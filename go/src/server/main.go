package main

import (
	"compress/gzip"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"server/db"
	"strconv"
	"strings"
	"time"

	"github.com/gin-contrib/multitemplate"
	"github.com/gin-gonic/gin"
)

func checkUser(c *gin.Context) (*db.User, uint64, error) {
	if len(c.PostForm("user")) == 0 {
		return nil, 0, errors.New("No user supplied")
	}
	if len(c.PostForm("user")) > 32 {
		return nil, 0, errors.New("Username too long")
	}

	user := &db.User{
		Password: c.PostForm("password"),
	}
	err := db.GetDB().Where(db.User{Username: c.PostForm("user")}).FirstOrCreate(&user).Error
	if err != nil {
		return nil, 0, err
	}

	// Ensure passwords match
	if user.Password != c.PostForm("password") {
		return nil, 0, errors.New("Incorrect password")
	}

	version, err := strconv.ParseUint(c.PostForm("version"), 10, 64)
	if err != nil {
		return nil, 0, errors.New("Invalid version")
	}
	if version < 7 {
		log.Println("Rejecting old game from %s, version %d", user.Username, version)
		return nil, 0, errors.New("\n\n\n\n\nYou must upgrade to a newer version!!\n\n\n\n\n")
	}

	return user, version, nil
}

func nextGame(c *gin.Context) {
	user, _, err := checkUser(c)
	if err != nil {
		log.Println(strings.TrimSpace(err.Error()))
		c.String(http.StatusBadRequest, err.Error())
		return
	}

	training_run := db.TrainingRun{
		Active: true,
	}
	// TODO(gary): Only really supports one training run right now...
	err = db.GetDB().Where(&training_run).First(&training_run).Error
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Invalid training run")
		return
	}

	network := db.Network{}
	err = db.GetDB().Where("id = ?", training_run.BestNetworkID).First(&network).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	if user != nil {
		var match []db.Match
		err = db.GetDB().Preload("Candidate").Where("done=false").Limit(1).Find(&match).Error
		if err != nil {
			log.Println(err)
			c.String(500, "Internal error")
			return
		}
		if len(match) > 0 {
			// Return this match
			match_game := db.MatchGame{
				UserID:  user.ID,
				MatchID: match[0].ID,
			}
			err = db.GetDB().Create(&match_game).Error
			// Note, this could cause an imbalance of white/black games for a particular match,
			// but it's good enough for now.
			flip := (match_game.ID & 1) == 1
			db.GetDB().Model(&match_game).Update("flip", flip)
			if err != nil {
				log.Println(err)
				c.String(500, "Internal error")
				return
			}
			result := gin.H{
				"type":         "match",
				"matchGameId":  match_game.ID,
				"sha":          network.Sha,
				"candidateSha": match[0].Candidate.Sha,
				"params":       match[0].Parameters,
				"flip":         flip,
			}
			c.JSON(http.StatusOK, result)
			return
		}
	}

	result := gin.H{
		"type":       "train",
		"trainingId": training_run.ID,
		"networkId":  training_run.BestNetworkID,
		"sha":        network.Sha,
		"params":     training_run.TrainParameters,
	}
	c.JSON(http.StatusOK, result)
}

// Computes SHA256 of gzip compressed file
func computeSha(http_file *multipart.FileHeader) (string, error) {
	h := sha256.New()
	file, err := http_file.Open()
	if err != nil {
		return "", err
	}
	defer file.Close()

	zr, err := gzip.NewReader(file)
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(h, zr); err != nil {
		return "", err
	}
	sha := fmt.Sprintf("%x", h.Sum(nil))
	if len(sha) != 64 {
		return "", errors.New("Hash length is not 64")
	}

	return sha, nil
}

func getTrainingRun(training_id uint) (*db.TrainingRun, error) {
	var training_run db.TrainingRun
	err := db.GetDB().Where("id = ?", training_id).First(&training_run).Error
	if err != nil {
		return nil, err
	}
	return &training_run, nil
}

func uploadNetwork(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		log.Println(err.Error())
		c.String(http.StatusBadRequest, "Missing file")
		return
	}

	// Compute hash of network
	sha, err := computeSha(file)
	if err != nil {
		log.Println(err.Error())
		c.String(500, "Internal error")
		return
	}
	network := db.Network{
		Sha: sha,
	}

	// Check for existing network
	var networkCount int
	err = db.GetDB().Model(&network).Where(&network).Count(&networkCount).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}
	if networkCount > 0 {
		c.String(http.StatusBadRequest, "Network already exists")
		return
	}

	// Create new network
	// TODO(gary): Just hardcoding this for now.
	var training_run_id uint = 1
	network.TrainingRunID = training_run_id
	layers, err := strconv.ParseInt(c.PostForm("layers"), 10, 32)
	network.Layers = int(layers)
	filters, err := strconv.ParseInt(c.PostForm("filters"), 10, 32)
	network.Filters = int(filters)
	err = db.GetDB().Create(&network).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}
	err = db.GetDB().Model(&network).Update("path", filepath.Join("networks", network.Sha)).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	os.MkdirAll(filepath.Dir(network.Path), os.ModePerm)

	// Save the file
	if err := c.SaveUploadedFile(file, network.Path); err != nil {
		log.Println(err.Error())
		c.String(500, "Saving file")
		return
	}

	// TODO(gary): Make this more generic - upload to s3 for now
	cmd := exec.Command("aws", "s3", "cp", network.Path, "s3://lczero/networks/")
	err = cmd.Run()
	if err != nil {
		log.Println(err.Error())
		c.String(500, "Uploading to s3")
		return
	}

	// Create a match to see if this network is better
	training_run, err := getTrainingRun(training_run_id)
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	match := db.Match{
		TrainingRunID: training_run_id,
		CandidateID:   network.ID,
		CurrentBestID: training_run.BestNetworkID,
		Done:          false,
		GameCap:       400,
		Parameters:    `["--tempdecay=10"]`,
	}
	err = db.GetDB().Create(&match).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	c.String(http.StatusOK, fmt.Sprintf("Network %s uploaded successfully.", network.Sha))
}

func uploadGame(c *gin.Context) {
	user, version, err := checkUser(c)
	if err != nil {
		log.Println(strings.TrimSpace(err.Error()))
		c.String(http.StatusBadRequest, err.Error())
		return
	}
	if version, err := strconv.ParseFloat(c.PostForm("engineVersion")[1:], 64); err != nil || version < 0.7 - 1e-6{
		log.Printf("Rejecting game with old lczero version %s", c.PostForm("engineVersion"))
		c.String(http.StatusBadRequest, "\n\n\n\n\nYou must upgrade to a newer lczero version!!\n\n\n\n\n")
		return
	}

	training_id, err := strconv.ParseUint(c.PostForm("training_id"), 10, 32)
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Invalid training_id")
	}

	training_run, err := getTrainingRun(uint(training_id))
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	network_id, err := strconv.ParseUint(c.PostForm("network_id"), 10, 32)
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Invalid network_id")
		return
	}

	var network db.Network
	err = db.GetDB().Where("id = ?", network_id).First(&network).Error
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Invalid network")
		return
	}

	err = db.GetDB().Exec("UPDATE networks SET games_played = games_played + 1 WHERE id = ?", network_id).Error
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Internal error")
		return
	}

	// Source
	file, err := c.FormFile("file")
	if err != nil {
		log.Println(err.Error())
		c.String(http.StatusBadRequest, "Missing file")
		return
	}

	// Create new game
	game := db.TrainingGame{
		UserID:        user.ID,
		TrainingRunID: training_run.ID,
		NetworkID:     network.ID,
		Version:       uint(version),
		Pgn:           c.PostForm("pgn"),
		EngineVersion: c.PostForm("engineVersion"),
	}
	db.GetDB().Create(&game)
	db.GetDB().Model(&game).Update("path", filepath.Join("games", fmt.Sprintf("run%d/training.%d.gz", training_run.ID, game.ID)))

	os.MkdirAll(filepath.Dir(game.Path), os.ModePerm)

	// Save the file
	if err := c.SaveUploadedFile(file, game.Path); err != nil {
		log.Println(err.Error())
		c.String(500, "Saving file")
		return
	}

	c.String(http.StatusOK, fmt.Sprintf("File %s uploaded successfully with fields user=%s.", file.Filename, user.Username))
}

func getNetwork(c *gin.Context) {
	network := db.Network{
		Sha: c.Query("sha"),
	}

	// Check for existing network
	err := db.GetDB().Where(&network).First(&network).Error
	if err != nil {
		log.Println(err)
		c.String(400, "Unknown network")
		return
	}

	// Serve the file
	// NOTE: Disabled due to bandwidth, re-enable this for tests...
	// c.File(network.Path)
	c.Redirect(http.StatusMovedPermanently, "https://s3.amazonaws.com/lczero/" + network.Path)
}

func setBestNetwork(training_id uint, network_id uint) error {
	// Set the best network of this training_run
	training_run, err := getTrainingRun(training_id)
	if err != nil {
		return err
	}
	err = db.GetDB().Model(&training_run).Update("best_network_id", network_id).Error
	if err != nil {
		return err
	}
	return nil
}

func checkMatchFinished(match_id uint) error {
	// Now check to see if match is finished
	var match db.Match
	err := db.GetDB().Where("id = ?", match_id).First(&match).Error
	if err != nil {
		return err
	}

	// Already done?  Just return
	if match.Done {
		return nil
	}

	if match.Wins+match.Losses+match.Draws >= match.GameCap {
		err = db.GetDB().Model(&match).Update("done", true).Error
		if err != nil {
			return err
		}
		if match.TestOnly {
			return nil
		}
		// Update to our new best network
		// TODO(SPRT)
		passed := calcElo(match.Wins, match.Losses, match.Draws) > -150.0
		err = db.GetDB().Model(&match).Update("passed", passed).Error
		if err != nil {
			return err
		}
		if passed {
			err = setBestNetwork(match.TrainingRunID, match.CandidateID)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func matchResult(c *gin.Context) {
	user, version, err := checkUser(c)
	if err != nil {
		log.Println(strings.TrimSpace(err.Error()))
		c.String(http.StatusBadRequest, err.Error())
		return
	}

	match_game_id, err := strconv.ParseUint(c.PostForm("match_game_id"), 10, 32)
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Invalid match_game_id")
		return
	}

	var match_game db.MatchGame
	err = db.GetDB().Where("id = ?", match_game_id).First(&match_game).Error
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Invalid match_game")
		return
	}

	result, err := strconv.ParseInt(c.PostForm("result"), 10, 32)
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Unable to parse result")
		return
	}

	good_result := result == 0 || result == -1 || result == 1
	if !good_result {
		c.String(http.StatusBadRequest, "Bad result")
		return
	}

	err = db.GetDB().Model(&match_game).Updates(db.MatchGame{
		Version:       uint(version),
		Result:        int(result),
		Done:          true,
		Pgn:           c.PostForm("pgn"),
		EngineVersion: c.PostForm("engineVersion"),
	}).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	col := ""
	if result == 0 {
		col = "draws"
	} else if result == 1 {
		col = "wins"
	} else {
		col = "losses"
	}
	// Atomic update of game count
	err = db.GetDB().Exec(fmt.Sprintf("UPDATE matches SET %s = %s + 1 WHERE id = ?", col, col), match_game.MatchID).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	err = checkMatchFinished(match_game.MatchID)
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	c.String(http.StatusOK, fmt.Sprintf("Match game %d successfuly uploaded from user=%s.", match_game.ID, user.Username))
}

func getActiveUsers() (gin.H, error) {
	rows, err := db.GetDB().Raw(`SELECT user_id, username, MAX(version), MAX(engine_version), MAX(training_games.created_at), count(*) FROM training_games
LEFT JOIN users
ON users.id = training_games.user_id
WHERE training_games.created_at >= now() - INTERVAL '1 day'
GROUP BY user_id, username
ORDER BY count DESC`).Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	active_users := 0
	games_played := 0
	users_json := []gin.H{}
	for rows.Next() {
		var user_id uint
		var username string
		var version int
		var engine_version string
		var created_at time.Time
		var count uint64
		rows.Scan(&user_id, &username, &version, &engine_version, &created_at, &count)

		active_users += 1
		games_played += int(count)

		if len(username) > 32 {
			username = username[0:32] + "..."
		}

		users_json = append(users_json, gin.H{
			"user":         username,
			"games_today":  count,
			"system":       "",
			"version":      version,
			"engine":       engine_version,
			"last_updated": created_at,
		})
	}

	result := gin.H{
		"active_users": active_users,
		"games_played": games_played,
		"users":        users_json,
	}
	return result, nil
}

func calcElo(wins int, losses int, draws int) float64 {
	score := float64(wins) + float64(draws)*0.5
	total := float64(wins + losses + draws)
	return -400 * math.Log10(1.0/(score/total)-1.0)
}

func getProgress() ([]gin.H, error) {
	var matches []db.Match
	err := db.GetDB().Order("id").Find(&matches).Error
	if err != nil {
		return nil, err
	}

	var networks []db.Network
	err = db.GetDB().Order("id").Find(&networks).Error
	if err != nil {
		return nil, err
	}

	counts := getNetworkCounts(networks)

	result := []gin.H{}
	result = append(result, gin.H{
		"net":    0,
		"rating": 0.0,
		"best":   false,
		"sprt":   "FAIL",
		"id":     "",
	})

	var count uint64 = 0
	var elo float64 = 0.0
	var matchIdx int = 0
	for _, network := range networks {
		var sprt string = "???"
		var best bool = false
		for matchIdx < len(matches) && matches[matchIdx].CandidateID == network.ID {
			if matches[matchIdx].TestOnly {
				matchIdx += 1
				continue
			}
			matchElo := calcElo(matches[matchIdx].Wins, matches[matchIdx].Losses, matches[matchIdx].Draws)
			if matches[matchIdx].Done {
				if matches[matchIdx].Passed {
					sprt = "PASS"
					best = true
				} else {
					sprt = "FAIL"
					best = false
				}
			}
			result = append(result, gin.H{
				"net":    count,
				"rating": elo + matchElo,
				"best":   best,
				"sprt":   sprt,
				"id":     network.ID,
			})
			if matches[matchIdx].Passed {
				elo += matchElo
			}
			matchIdx += 1
		}
		// TODO(gary): Hack for start...
		if network.ID == 3 {
			result = append(result, gin.H{
				"net":    count,
				"rating": elo,
				"best":   true,
				"sprt":   sprt,
				"id":     network.ID,
			})
		}
		count += counts[network.ID]
	}

	return result, nil
}

func filterProgress(result []gin.H) []gin.H {
	// Show just the last 100 networks
	if len(result) > 100 {
		result = result[len(result)-100:]
	}

	// Ensure the ordering is correct now (HACK)
	tmp := []gin.H{}
	tmp = append(tmp, gin.H{
		"net":    result[0]["net"],
		"rating": result[0]["rating"],
		"best":   false,
		"sprt":   "???",
		"id":     "",
	})
	tmp = append(tmp, gin.H{
		"net":    result[0]["net"],
		"rating": result[0]["rating"],
		"best":   false,
		"sprt":   "FAIL",
		"id":     "",
	})

	return append(tmp, result...)
}

func frontPage(c *gin.Context) {
	users, err := getActiveUsers()
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	progress, err := getProgress()
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}
	if c.DefaultQuery("full_elo", "0") == "0" {
		progress = filterProgress(progress)
	}

	/*
	c.HTML(http.StatusOK, "index", gin.H{
		"active_users": 0,
		"games_played": 0,
		"Users":        []gin.H{},
		"progress":     progress,
	})
	*/
	c.HTML(http.StatusOK, "index", gin.H{
		"active_users": users["active_users"],
		"games_played": users["games_played"],
		"Users":        users["users"],
		"progress":     progress,
	})
}

func user(c *gin.Context) {
	// TODO(gary): Optimize this!
	c.HTML(http.StatusOK, "user", gin.H{
		"user":  c.Param("name"),
		"games": []gin.H{},
	})
	return

	name := c.Param("name")
	user := db.User{
		Username: name,
	}
	err := db.GetDB().Where(&user).First(&user).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	games := []db.TrainingGame{}
	err = db.GetDB().Model(&user).Preload("Network").Limit(50).Order("created_at DESC").Related(&games).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	gamesJson := []gin.H{}
	for _, game := range games {
		gamesJson = append(gamesJson, gin.H{
			"id":         game.ID,
			"created_at": game.CreatedAt.String(),
			"network":    game.Network.Sha,
		})
	}

	c.HTML(http.StatusOK, "user", gin.H{
		"user":  user.Username,
		"games": gamesJson,
	})
}

func game(c *gin.Context) {
	c.HTML(http.StatusOK, "game", gin.H{
		"pgn": "Disabled for now -- server load too high from scaping",
	})
	return

	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	game := db.TrainingGame{
		ID: uint64(id),
	}
	err = db.GetDB().Where(&game).First(&game).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	c.HTML(http.StatusOK, "game", gin.H{
		"pgn": game.Pgn,
	})
}

func viewMatchGame(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	game := db.MatchGame{
		ID: uint64(id),
	}
	err = db.GetDB().Where(&game).First(&game).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	c.HTML(http.StatusOK, "game", gin.H{
		"pgn": strings.Replace(game.Pgn, "e.p.", "", -1),
	})
}

func getNetworkCounts(networks []db.Network) map[uint]uint64 {
	counts := make(map[uint]uint64)
	for _, network := range networks {
		counts[network.ID] = uint64(network.GamesPlayed)
	}
	return counts
}

func viewNetworks(c *gin.Context) {
	// TODO(gary): Whole thing needs to take training_run into account...
	var networks []db.Network
	err := db.GetDB().Order("id desc").Find(&networks).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	progress, err := getProgress()
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	counts := getNetworkCounts(networks)
	json := []gin.H{}
	for i, network := range networks {
		json = append(json, gin.H{
			"id":         network.ID,
			"elo":        fmt.Sprintf("%.2f", progress[len(progress)-1-i]["rating"]),
			"games":      counts[network.ID],
			"sha":        network.Sha,
			"short_sha":  network.Sha[0:8],
			"blocks":     network.Layers,
			"filters":    network.Filters,
			"created_at": network.CreatedAt,
		})
	}

	c.HTML(http.StatusOK, "networks", gin.H{
		"networks": json,
	})
}

func viewTrainingRuns(c *gin.Context) {
	training_runs := []db.TrainingRun{}
	err := db.GetDB().Find(&training_runs).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	rows := []gin.H{}
	for _, training_run := range training_runs {
		rows = append(rows, gin.H{
			"id":            training_run.ID,
			"active":        training_run.Active,
			"trainParams":   training_run.TrainParameters,
			"bestNetworkId": training_run.BestNetworkID,
			"description":   training_run.Description,
		})
	}

	c.HTML(http.StatusOK, "training_runs", gin.H{
		"training_runs": rows,
	})
}

func viewStats(c *gin.Context) {
	var networks []db.Network
	err := db.GetDB().Order("id desc").Where("games_played > 0").Limit(3).Find(&networks).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	json := []gin.H{}
	for _, network := range networks {
		json = append(json, gin.H{
			"short_sha": network.Sha[0:8],
		})
	}

	c.HTML(http.StatusOK, "stats", gin.H{
		"networks": json,
	})
}

func viewMatches(c *gin.Context) {
	var matches []db.Match
	err := db.GetDB().Order("id desc").Find(&matches).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	json := []gin.H{}
	for _, match := range matches {
		elo := calcElo(match.Wins, match.Losses, match.Draws)
		table_class := "active"
		if match.Done {
			if match.Passed {
				table_class = "success"
			} else {
				table_class = "danger"
			}
		}
		json = append(json, gin.H{
			"id":           match.ID,
			"current_id":   match.CurrentBestID,
			"candidate_id": match.CandidateID,
			"score":        fmt.Sprintf("+%d -%d =%d", match.Wins, match.Losses, match.Draws),
			"elo":          fmt.Sprintf("%.2f", elo),
			"done":         match.Done,
			"table_class":  table_class,
			"passed":       match.Passed,
			"created_at":   match.CreatedAt,
		})
	}

	c.HTML(http.StatusOK, "matches", gin.H{
		"matches": json,
	})
}

func viewMatch(c *gin.Context) {
	match := db.Match{}
	err := db.GetDB().Where("id = ?", c.Param("id")).First(&match).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	games := []db.MatchGame{}
	err = db.GetDB().Where(&db.MatchGame{MatchID: match.ID}).Preload("User").Order("id").Find(&games).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	gamesJson := []gin.H{}
	for _, game := range games {
		color := "white"
		if game.Flip {
			color = "black"
		}
		result := "-"
		if game.Done {
			if game.Result == 1 {
				result = "win"
			} else if game.Result == -1 {
				result = "loss"
			} else {
				result = "draw"
			}
		}
		gamesJson = append(gamesJson, gin.H{
			"id":         game.ID,
			"created_at": game.CreatedAt.String(),
			"result":     result,
			"done":       game.Done,
			"user":       game.User.Username,
			"color":      color,
		})
	}

	c.HTML(http.StatusOK, "match", gin.H{
		"games": gamesJson,
	})
}

func viewTrainingData(c *gin.Context) {
	rows, err := db.GetDB().Raw(`SELECT MAX(id) FROM training_games WHERE compacted = true`).Rows()
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}
	defer rows.Close()

	var id uint
	for rows.Next() {
		rows.Scan(&id)
		break
	}

	files := []gin.H{}
	game_id := uint(30000)
	for game_id < id {
		files = append([]gin.H{
			gin.H{"url": fmt.Sprintf("https://s3.amazonaws.com/lczero/training/games%d.tar.gz", game_id)},
		}, files...)
		game_id += 10000
	}

	c.HTML(http.StatusOK, "training_data", gin.H{
		"files": files,
	})
}

func createTemplates() multitemplate.Render {
	r := multitemplate.New()
	r.AddFromFiles("index", "templates/base.tmpl", "templates/index.tmpl")
	r.AddFromFiles("user", "templates/base.tmpl", "templates/user.tmpl")
	r.AddFromFiles("game", "templates/base.tmpl", "templates/game.tmpl")
	r.AddFromFiles("networks", "templates/base.tmpl", "templates/networks.tmpl")
	r.AddFromFiles("training_runs", "templates/base.tmpl", "templates/training_runs.tmpl")
	r.AddFromFiles("stats", "templates/base.tmpl", "templates/stats.tmpl")
	r.AddFromFiles("match", "templates/base.tmpl", "templates/match.tmpl")
	r.AddFromFiles("matches", "templates/base.tmpl", "templates/matches.tmpl")
	r.AddFromFiles("training_data", "templates/base.tmpl", "templates/training_data.tmpl")
	return r
}

func setupRouter() *gin.Engine {
	router := gin.Default()
	router.HTMLRender = createTemplates()
	router.MaxMultipartMemory = 32 << 20 // 32 MiB
	router.Static("/css", "./public/css")
	router.Static("/js", "./public/js")
	router.Static("/stats", "/home/web/netstats")

	router.GET("/", frontPage)
	router.GET("/get_network", getNetwork)
	router.GET("/user/:name", user)
	router.GET("/game/:id", game)
	router.GET("/networks", viewNetworks)
	router.GET("/stats", viewStats)
	router.GET("/training_runs", viewTrainingRuns)
	router.GET("/match/:id", viewMatch)
	router.GET("/matches", viewMatches)
	router.GET("/match_game/:id", viewMatchGame)
	router.GET("/training_data", viewTrainingData)
	router.POST("/next_game", nextGame)
	router.POST("/upload_game", uploadGame)
	router.POST("/upload_network", uploadNetwork)
	router.POST("/match_result", matchResult)
	return router
}

func main() {
	db.Init(true)
	db.SetupDB()
	defer db.Close()

	router := setupRouter()
	router.Run(":8080")
}
