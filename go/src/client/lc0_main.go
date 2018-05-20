// A new client to work with the lc0 binary.
//
//
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"client/http"

	"github.com/Tilps/chess"
)

var hostname = flag.String("hostname", "http://api.lczero.org", "Address of the server")
var user = flag.String("user", "", "Username")
var password = flag.String("password", "", "Password")
var gpu = flag.Int("gpu", -1, "ID of the OpenCL device to use (-1 for default, or no GPU)")
var debug = flag.Bool("debug", false, "Enable debug mode to see verbose output and save logs")

type Settings struct {
	User string
	Pass string
}

//var lc0_args = []string{"--temperature=1", "--noise", "--no-smart-pruning", "--minibatch-size=1"}
var lc0_args = []string{"--temperature=1", "--noise"}

/*
	Reads the user and password from a config file and returns empty strings if anything went wrong.
	If the config file does not exists, it prompts the user for a username and password and creates the config file.
*/
func readSettings(path string) (string, string) {
	settings := Settings{}
	file, err := os.Open(path)
	if err != nil {
		// File was not found
		fmt.Printf("Please enter your username and password, an account will be automatically created.\n")
		fmt.Printf("Note that this password will be stored in plain text, so avoid a password that is\n")
		fmt.Printf("also used for sensitive applications. It also cannot be recovered.\n")
		fmt.Printf("Enter username : ")
		fmt.Scanf("%s\n", &settings.User)
		fmt.Printf("Enter password : ")
		fmt.Scanf("%s\n", &settings.Pass)
		jsonSettings, err := json.Marshal(settings)
		if err != nil {
			log.Fatal("Cannot encode settings to JSON ", err)
			return "", ""
		}
		settingsFile, err := os.Create(path)
		if err != nil {
			log.Fatal("Could not create output file ", err)
			return "", ""
		}
		fmt.Fprintf(settingsFile, "%s", jsonSettings)
		return settings.User, settings.Pass
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&settings)
	if err != nil {
		log.Fatal("Error decoding JSON ", err)
		return "", ""
	}
	return settings.User, settings.Pass
}

func getExtraParams() map[string]string {
	return map[string]string{
		"user":     *user,
		"password": *password,
		"version":  "10",
	}
}

func uploadGame(httpClient *http.Client, path string, pgn string, nextGame client.NextGameResponse, version string, retryCount uint) error {
	//	return nil

	extraParams := getExtraParams()
	extraParams["training_id"] = strconv.Itoa(int(nextGame.TrainingId))
	extraParams["network_id"] = strconv.Itoa(int(nextGame.NetworkId))
	extraParams["pgn"] = pgn
	extraParams["engineVersion"] = version
	request, err := client.BuildUploadRequest(*hostname+"/upload_game", extraParams, "file", path)
	if err != nil {
		log.Printf("BUR: %v", err)
		return err
	}
	resp, err := httpClient.Do(request)
	if err != nil {
		log.Printf("http.Do: %v", err)
		return err
	}
	body := &bytes.Buffer{}
	_, err = body.ReadFrom(resp.Body)
	if err != nil {
		log.Print(err)
		log.Print("Error uploading, retrying...")
		time.Sleep(time.Second * (2 << retryCount))
		err = uploadGame(httpClient, path, pgn, nextGame, version, retryCount+1)
		return err
	}
	resp.Body.Close()
	fmt.Println(resp.StatusCode)
	fmt.Println(resp.Header)
	fmt.Println(body)

	train_dir := filepath.Dir(path)
	if _, err := os.Stat(train_dir); err == nil {
		files, err := ioutil.ReadDir(train_dir)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Cleanup training files:\n")
		for _, f := range files {
			fmt.Printf("%s/%s\n", train_dir, f.Name())
		}
		err = os.RemoveAll(train_dir)
		if err != nil {
			log.Fatal(err)
		}
	}

	return nil
}

type gameInfo struct {
	pgn   string
	fname string
}

type CmdWrapper struct {
	Cmd      *exec.Cmd
	Pgn      string
	Input    io.WriteCloser
	BestMove chan string
	gi       chan gameInfo
	Version  string
}

func (c *CmdWrapper) openInput() {
	var err error
	c.Input, err = c.Cmd.StdinPipe()
	if err != nil {
		log.Fatal(err)
	}
}

func convertMovesToPGN(moves []string) string {
	game := chess.NewGame(chess.UseNotation(chess.LongAlgebraicNotation{}))
	for _, m := range moves {
		err := game.MoveStr(m)
		if err != nil {
			log.Fatalf("movstr: %v", err)
		}
	}
	return game.String()
}

func CreateCmdWrapper() *CmdWrapper {
	c := &CmdWrapper{
		gi:       make(chan gameInfo),
		BestMove: make(chan string),
	}
	return c
}

func (c *CmdWrapper) launch(networkPath string, args []string, input bool) {
	weights := fmt.Sprintf("--weights=%s", networkPath)
	dir, _ := os.Getwd()
	lcargs := []string{"selfplay", weights, "--training=true", "--visits=800"}
	lcargs = append(lcargs, lc0_args...)
	c.Cmd = exec.Command(path.Join(dir, "lc0"), lcargs...)
	// TODO: server args are hosed right now.
	//	c.Cmd.Args = append(c.Cmd.Args, args...)
	if *gpu != -1 {
		c.Cmd.Args = append(c.Cmd.Args, fmt.Sprintf("--gpu=%v", *gpu))
	}
	if !*debug {
		//		c.Cmd.Args = append(c.Cmd.Args, "--quiet")
		fmt.Println("lc0 is never quiet.")
	}
	fmt.Printf("Args: %v\n", c.Cmd.Args)

	stdout, err := c.Cmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
	}

	stderr, err := c.Cmd.StderrPipe()
	if err != nil {
		log.Fatal(err)
	}

	go func() {
		defer close(c.BestMove)
		defer close(c.gi)
		stdoutScanner := bufio.NewScanner(stdout)
		for stdoutScanner.Scan() {
			line := stdoutScanner.Text()
			//			fmt.Printf("lc0: %s\n", line)
			switch {
			case strings.HasPrefix(line, "gameready"):
				parts := strings.Split(line, " ")
				if parts[1] != "trainingfile" ||
					parts[3] != "gameid" ||
					parts[5] != "player1" ||
					parts[7] != "result" ||
					parts[9] != "moves" {
					log.Printf("Malformed gameready: %q", line)
					break
				}
				file := parts[2]
				//gameid := parts[4]
				//result := parts[8]
				pgn := convertMovesToPGN(parts[10:])
				fmt.Printf("PGN: %s\n", pgn)
				c.gi <- gameInfo{pgn: pgn, fname: file}
			case strings.HasPrefix(line, "bestmove "):
				//				c.BestMove <- strings.Split(line, " ")[1]
			case strings.HasPrefix(line, "id name lczero "):
				c.Version = strings.Split(line, " ")[3]
			case strings.HasPrefix(line, "info"):
				// just swallow this line.
			default:
				fmt.Println(line)
			}
		}
	}()

	go func() {
		stderrScanner := bufio.NewScanner(stderr)
		for stderrScanner.Scan() {
			fmt.Printf("%s\n", stderrScanner.Text())
		}
	}()

	if input {
		c.openInput()
	}

	err = c.Cmd.Start()
	if err != nil {
		log.Fatal(err)
	}
}

func playMatch(baselinePath string, candidatePath string, params []string, flip bool) (int, string, string, error) {
	baseline := CmdWrapper{}
	baseline.launch(baselinePath, params, true)
	defer baseline.Input.Close()

	candidate := CmdWrapper{}
	candidate.launch(candidatePath, params, true)
	defer candidate.Input.Close()

	p1 := &candidate
	p2 := &baseline

	if flip {
		p2, p1 = p1, p2
	}

	io.WriteString(baseline.Input, "uci\n")
	io.WriteString(candidate.Input, "uci\n")

	// Play a game using UCI
	var result int
	game := chess.NewGame(chess.UseNotation(chess.LongAlgebraicNotation{}))
	move_history := ""
	turn := 0
	for {
		if turn >= 450 || game.Outcome() != chess.NoOutcome || len(game.EligibleDraws()) > 1 {
			if game.Outcome() == chess.WhiteWon {
				result = 1
			} else if game.Outcome() == chess.BlackWon {
				result = -1
			} else {
				result = 0
			}

			// Always report the result relative to the candidate engine (which defaults to white, unless flip = true)
			if flip {
				result = -result
			}
			break
		}

		var p *CmdWrapper
		if game.Position().Turn() == chess.White {
			p = p1
		} else {
			p = p2
		}
		io.WriteString(p.Input, "position startpos"+move_history+"\n")
		io.WriteString(p.Input, "go\n")

		select {
		case best_move, ok := <-p.BestMove:
			if !ok {
				p.BestMove = nil
				break
			}
			err := game.MoveStr(best_move)
			if err != nil {
				log.Println("Error decoding: " + best_move + " for game:\n" + game.String())
				return 0, "", "", err
			}
			if len(move_history) == 0 {
				move_history = " moves"
			}
			move_history += " " + best_move
			turn += 1
		case <-time.After(60 * time.Second):
			log.Println("Bestmove has timed out, aborting match")
			return 0, "", "", errors.New("timeout")
		}
	}

	chess.UseNotation(chess.AlgebraicNotation{})(game)
	return result, game.String(), candidate.Version, nil
}

func train(httpClient *http.Client, ngr client.NextGameResponse,
	networkPath string, count int, params []string, doneCh chan bool) {
	// pid is intended for use in multi-threaded training
	pid := os.Getpid()

	dir, _ := os.Getwd()
	//	train_dir := path.Join(dir, fmt.Sprintf("data-%v-%v", pid, count))
	if *debug {
		logs_dir := path.Join(dir, fmt.Sprintf("logs-%v", pid))
		os.MkdirAll(logs_dir, os.ModePerm)
		logfile := path.Join(logs_dir, fmt.Sprintf("%s.log", time.Now().Format("20060102150405")))
		params = append(params, "-l"+logfile)
	}

	num_games := 1
	train_cmd := fmt.Sprintf("--start=train %v-%v %v", pid, count, num_games)
	params = append(params, train_cmd)

	c := CreateCmdWrapper()
	c.Version = "v0.10"
	c.launch(networkPath, params, false)
	for done := false; !done; {
		select {
		case <-doneCh:
			done = true
			c.Cmd.Process.Kill()
			break
		case gi, ok := <-c.gi:
			if !ok {
				done = true
				break
			}
			fmt.Printf("Uploading game: %d\n", num_games)
			num_games++
			go uploadGame(httpClient, gi.fname, gi.pgn, ngr, c.Version, 0)
		}
	}

	err := c.Cmd.Wait()
	if err != nil {
		log.Fatal(err)
	}

	return
}

func getNetwork(httpClient *http.Client, sha string, clearOld bool) (string, error) {
	// Sha already exists?
	path := filepath.Join("networks", sha)
	if stat, err := os.Stat(path); err == nil {
		if stat.Size() != 0 {
			return path, nil
		}
	}

	if clearOld {
		// Clean out any old networks
		os.RemoveAll("networks")
	}
	os.MkdirAll("networks", os.ModePerm)

	fmt.Printf("Downloading network...\n")
	// Otherwise, let's download it
	err := client.DownloadNetwork(httpClient, *hostname, path, sha)
	if err != nil {
		return "", err
	}
	return path, nil
}

func nextGame(httpClient *http.Client, count int) error {
	nextGame, err := client.NextGame(httpClient, *hostname, getExtraParams())
	if err != nil {
		return err
	}
	var params []string
	err = json.Unmarshal([]byte(nextGame.Params), &params)
	if err != nil {
		return err
	}

	if nextGame.Type == "match" {
		networkPath, err := getNetwork(httpClient, nextGame.Sha, false)
		if err != nil {
			return err
		}
		candidatePath, err := getNetwork(httpClient, nextGame.CandidateSha, false)
		if err != nil {
			return err
		}
		result, pgn, version, err := playMatch(networkPath, candidatePath, params, nextGame.Flip)
		if err != nil {
			return err
		}
		extraParams := getExtraParams()
		extraParams["engineVersion"] = version
		go client.UploadMatchResult(httpClient, *hostname, nextGame.MatchGameId, result, pgn, extraParams)
		return nil
	}

	if nextGame.Type == "train" {
		networkPath, err := getNetwork(httpClient, nextGame.Sha, true)
		if err != nil {
			return err
		}
		ch := make(chan bool)
		go func() {
			errCount := 0
			for {
				time.Sleep(60 * time.Second)
				ng, err := client.NextGame(httpClient, *hostname, getExtraParams())
				if err != nil {
					fmt.Printf("Error talking to server: %v\n", err)
					errCount++
					if errCount < 10 {
						continue
					}
				}
				if err != nil || ng.Type != nextGame.Type || ng.Sha != nextGame.Sha {
					ch <- true
					close(ch)
					return
				}
				errCount = 0
			}
		}()
		train(httpClient, nextGame, networkPath, count, params, ch)
		return nil
	}

	return errors.New("Unknown game type: " + nextGame.Type)
}

func main() {
	flag.Parse()

	if len(*user) == 0 || len(*password) == 0 {
		*user, *password = readSettings("settings.json")
	}

	if len(*user) == 0 {
		log.Fatal("You must specify a username")
	}
	if len(*password) == 0 {
		log.Fatal("You must specify a non-empty password")
	}

	httpClient := &http.Client{}
	start := time.Now()
	for i := 0; ; i++ {
		err := nextGame(httpClient, i)
		if err != nil {
			log.Print(err)
			log.Print("Sleeping for 30 seconds...")
			time.Sleep(30 * time.Second)
			continue
		}
		elapsed := time.Since(start)
		log.Printf("Completed %d games in %s time", i+1, elapsed)
	}
}
