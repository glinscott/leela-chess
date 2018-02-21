package main

import (
	"compress/gzip"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"server/db"
	"strconv"

	"github.com/gin-contrib/multitemplate"
	"github.com/gin-gonic/gin"
)

func nextGame(c *gin.Context) {
	var training_run db.TrainingRun
	// TODO(gary): Need to set some sort of priority system here.
	err := db.GetDB().Preload("BestNetwork").First(&training_run).Error
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Invalid training run")
		return
	}

	// TODO: Check for active matches.

	result := gin.H{
		"type":       "train",
		"trainingId": training_run.ID,
		"networkId":  training_run.BestNetwork.ID,
		"sha":        training_run.BestNetwork.Sha,
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

func getTrainingRun(training_id string) (*db.TrainingRun, error) {
	id, err := strconv.ParseUint(training_id, 10, 32)
	if err != nil {
		return nil, err
	}

	var training_run db.TrainingRun
	err = db.GetDB().Where("id = ?", id).First(&training_run).Error
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

	// Set the best network of this training_run
	training_run, err := getTrainingRun(c.PostForm("training_id"))
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Invalid training run")
		return
	}
	training_run.BestNetwork = network
	err = db.GetDB().Save(training_run).Error
	if err != nil {
		log.Println(err)
		c.String(500, "Failed to update best training_run")
		return
	}

	c.String(http.StatusOK, fmt.Sprintf("Network %s uploaded successfully.", network.Sha))
}

func uploadGame(c *gin.Context) {
	var user db.User
	user.Password = c.PostForm("password")
	err := db.GetDB().Where(db.User{Username: c.PostForm("user")}).FirstOrCreate(&user).Error
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Invalid user")
		return
	}

	// Ensure passwords match
	if user.Password != c.PostForm("password") {
		c.String(http.StatusBadRequest, "Incorrect password")
		return
	}

	training_run, err := getTrainingRun(c.PostForm("training_id"))
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Invalid training run")
		return
	}

	var network db.Network
	err = db.GetDB().Where("id = ?", c.PostForm("network_id")).First(&network).Error
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "Invalid network")
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
	version, err := strconv.ParseUint(c.PostForm("version"), 10, 64)
	if err != nil {
		log.Println(err.Error())
		c.String(http.StatusBadRequest, "Invalid version")
		return
	}
	game := db.TrainingGame{
		User:        user,
		TrainingRun: *training_run,
		Network:     network,
		Version:     uint(version),
		Pgn:         c.PostForm("pgn"),
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
		Sha: c.Param("sha"),
	}

	// Check for existing network
	err := db.GetDB().Where(&network).First(&network).Error
	if err != nil {
		log.Println(err)
		c.String(400, "Unknown network")
		return
	}

	// Serve the file
	c.File(network.Path)
}

func getActiveUsers() ([]gin.H, error) {
	users := []db.User{}
	err := db.GetDB().Find(&users).Error
	if err != nil {
		return nil, err
	}

	result := []gin.H{}
	for _, user := range users {
		result = append(result, gin.H{
			"user":         user.Username,
			"games_hour":   0,
			"system":       "",
			"version":      "",
			"last_updated": user.UpdatedAt,
		})
	}
	return result, nil
}

func frontPage(c *gin.Context) {
	users, err := getActiveUsers()
	if err != nil {
		log.Println(err)
		c.String(500, "Internal error")
		return
	}

	c.HTML(http.StatusOK, "index", gin.H{
		"Users": users,
	})
}

func user(c *gin.Context) {
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

func createTemplates() multitemplate.Render {
	r := multitemplate.New()
	r.AddFromFiles("index", "templates/base.tmpl", "templates/index.tmpl")
	r.AddFromFiles("user", "templates/base.tmpl", "templates/user.tmpl")
	return r
}

func setupRouter() *gin.Engine {
	router := gin.Default()
	router.HTMLRender = createTemplates()
	router.MaxMultipartMemory = 32 << 20 // 32 MiB
	router.Static("/css", "./public/css")
	router.Static("/js", "./public/js")

	router.GET("/", frontPage)
	router.GET("/get_network", getNetwork)
	router.GET("/user/:name", user)
	router.POST("/next_game", nextGame)
	router.POST("/upload_game", uploadGame)
	router.POST("/upload_network", uploadNetwork)
	return router
}

func main() {
	db.Init(true)
	db.SetupDB()
	defer db.Close()

	router := setupRouter()
	router.Run(":8080")
}
