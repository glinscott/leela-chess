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

var hostname = flag.String("hostname", "http://testserver.lczero.org", "Address of the server")
var user = flag.String("user", "", "Username")
var password = flag.String("password", "", "Password")
var gpu = flag.Int("gpu", -1, "ID of the OpenCL device to use (-1 for default, or no GPU)")
var debug = flag.Bool("debug", false, "Enable debug mode to see verbose output and save logs")
var lc0Args = flag.String("lc0args", "", `Extra args to pass to the backend.  example: --lc0args="--parallelism=10 --threads=2"`)

// Settings holds username and password.
type Settings struct {
	User string
	Pass string
}

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
		defer settingsFile.Close()
		if err != nil {
			log.Fatal("Could not create output file ", err)
			return "", ""
		}
		settingsFile.Write(jsonSettings)
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

func uploadGame(httpClient *http.Client, path string, pgn string,
	nextGame client.NextGameResponse, version string, retryCount uint) error {
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

	trainDir := filepath.Dir(path)
	if _, err := os.Stat(trainDir); err == nil {
		files, err := ioutil.ReadDir(trainDir)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Cleanup training files:\n")
		for _, f := range files {
			fmt.Printf("%s/%s\n", trainDir, f.Name())
		}
		err = os.RemoveAll(trainDir)
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

type cmdWrapper struct {
	Cmd      *exec.Cmd
	Pgn      string
	Input    io.WriteCloser
	BestMove chan string
	gi       chan gameInfo
	Version  string
}

func (c *cmdWrapper) openInput() {
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
	game2 := chess.NewGame()
	b, err := game.MarshalText()
	if err != nil {
		log.Fatalf("MarshalText failed: %v", err)
	}
	game2.UnmarshalText(b)
	return game2.String()
}

func createCmdWrapper() *cmdWrapper {
	c := &cmdWrapper{
		gi:       make(chan gameInfo),
		BestMove: make(chan string),
	}
	return c
}

func (c *cmdWrapper) launch(networkPath string, args []string, input bool) {
	dir, _ := os.Getwd()
	c.Cmd = exec.Command(path.Join(dir, "lc0"))
	c.Cmd.Args = append(c.Cmd.Args, args...)
	c.Cmd.Args = append(c.Cmd.Args, fmt.Sprintf("--weights=%s", networkPath))
	if *lc0Args != "" {
		// TODO: We might want to inspect these to prevent someone
		// from passing a different visits or batch size for example.
		// Possibly just exposts exact args we want to passthrough like
		// backend.  For testing right now this is probably useful to not
		// need to rebuild the client to change lc0 args.
		parts := strings.Split(*lc0Args, " ")
		c.Cmd.Args = append(c.Cmd.Args, parts...)
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
				fmt.Println(line)
				c.BestMove <- strings.Split(line, " ")[1]
			case strings.HasPrefix(line, "id name lczero "):
				c.Version = strings.Split(line, " ")[3]
			case strings.HasPrefix(line, "info"):
				break
				fallthrough
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
	baseline := createCmdWrapper()
	params = append([]string{"uci"}, params...)
	log.Println("launching 1")
	baseline.launch(baselinePath, params, true)
	defer baseline.Input.Close()

	candidate := createCmdWrapper()
	log.Println("launching 2")
	candidate.launch(candidatePath, params, true)
	defer candidate.Input.Close()

	p1 := candidate
	p2 := baseline

	if flip {
		p2, p1 = p1, p2
	}

	log.Println("writign uci")
	io.WriteString(baseline.Input, "uci\n")
	io.WriteString(candidate.Input, "uci\n")

	// Play a game using UCI
	var result int
	game := chess.NewGame(chess.UseNotation(chess.LongAlgebraicNotation{}))
	moveHistory := ""
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

			// Always report the result relative to the candidate engine
			// (which defaults to white, unless flip = true)
			if flip {
				result = -result
			}
			log.Printf("result: %d\n", result)
			break
		}

		var p *cmdWrapper
		if game.Position().Turn() == chess.White {
			p = p1
		} else {
			p = p2
		}
		io.WriteString(p.Input, "position startpos"+moveHistory+"\n")
		io.WriteString(p.Input, "go nodes 800\n")
		log.Println("sent go")

		select {
		case bestMove, ok := <-p.BestMove:
			if !ok {
				log.Println("engine failed")
				p.BestMove = nil
				break
			}
			err := game.MoveStr(bestMove)
			if err != nil {
				log.Println("Error decoding: " + bestMove + " for game:\n" + game.String())
				return 0, "", "", err
			}
			if len(moveHistory) == 0 {
				moveHistory = " moves"
			}
			moveHistory += " " + bestMove
			turn++
		case <-time.After(60 * time.Second):
			log.Println("Bestmove has timed out, aborting match")
			return 0, "", "", errors.New("timeout")
		}
	}

	chess.UseNotation(chess.AlgebraicNotation{})(game)
	fmt.Printf("PGN: %s\n", game.String())
	return result, game.String(), candidate.Version, nil
}

func train(httpClient *http.Client, ngr client.NextGameResponse,
	networkPath string, count int, params []string, doneCh chan bool) {
	// pid is intended for use in multi-threaded training
	pid := os.Getpid()

	dir, _ := os.Getwd()
	if *debug {
		logsDir := path.Join(dir, fmt.Sprintf("logs-%v", pid))
		os.MkdirAll(logsDir, os.ModePerm)
		logfile := path.Join(logsDir, fmt.Sprintf("%s.log", time.Now().Format("20060102150405")))
		params = append(params, "-l"+logfile)
	}

	// lc0 needs selfplay first in the argument list.
	params = append([]string{"selfplay"}, params...)
	params = append(params, "--training=true")
	c := createCmdWrapper()
	c.Version = "v0.10"
	c.launch(networkPath, params /* input= */, false)
	for done := false; !done; {
		numGames := 1
		select {
		case <-doneCh:
			done = true
			log.Println("Received message to end training, killing lc0")
			c.Cmd.Process.Kill()
		case _, ok := <-c.BestMove:
			// Just swallow the best moves, only needed for match play.
			if !ok {
				log.Printf("BestMove channel closed unexpectedly, exiting train loop")
				break
			}
		case gi, ok := <-c.gi:
			if !ok {
				log.Printf("GameInfo channel closed, exiting train loop")
				done = true
				break
			}
			fmt.Printf("Uploading game: %d\n", numGames)
			numGames++
			go uploadGame(httpClient, gi.fname, gi.pgn, ngr, c.Version, 0)
		}
	}

	log.Println("Waiting for lc0 to stop")
	err := c.Cmd.Wait()
	if err != nil {
		log.Fatal(err)
	}
	log.Println("lc0 stopped")
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
		log.Printf("Network download failed: %v", err)
		return "", err
	}
	return path, nil
}

func validateParams(args []string) []string {
	validArgs := []string{}
	for _, arg := range args {
		if strings.HasPrefix(arg, "--tempdecay") {
			continue
		}
		validArgs = append(validArgs, arg)
	}
	return validArgs
}

func nextGame(httpClient *http.Client, count int) error {
	nextGame, err := client.NextGame(httpClient, *hostname, getExtraParams())
	if err != nil {
		return err
	}
	var serverParams []string
	err = json.Unmarshal([]byte(nextGame.Params), &serverParams)
	if err != nil {
		return err
	}
	log.Printf("serverParams: %s", serverParams)
	serverParams = validateParams(serverParams)

	if nextGame.Type == "match" {
		log.Println("Starting match")
		networkPath, err := getNetwork(httpClient, nextGame.Sha, false)
		if err != nil {
			return err
		}
		candidatePath, err := getNetwork(httpClient, nextGame.CandidateSha, false)
		if err != nil {
			return err
		}
		log.Println("Starting match")
		result, pgn, version, err := playMatch(networkPath, candidatePath, serverParams, nextGame.Flip)
		if err != nil {
			log.Fatalf("playMatch: %v", err)
			return err
		}
		extraParams := getExtraParams()
		extraParams["engineVersion"] = version
		log.Println("uploading match result")
		go client.UploadMatchResult(httpClient, *hostname, nextGame.MatchGameId, result, pgn, extraParams)
		return nil
	}

	if nextGame.Type == "train" {
		networkPath, err := getNetwork(httpClient, nextGame.Sha, true)
		if err != nil {
			return err
		}
		doneCh := make(chan bool)
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
					doneCh <- true
					close(doneCh)
					return
				}
				errCount = 0
			}
		}()
		train(httpClient, nextGame, networkPath, count, serverParams, doneCh)
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
//	start := time.Now()
	for i := 0; ; i++ {
		err := nextGame(httpClient, i)
		if err != nil {
			log.Print(err)
			log.Print("Sleeping for 30 seconds...")
			time.Sleep(30 * time.Second)
			continue
		}
//		elapsed := time.Since(start)
//		log.Printf("Completed %d games in %s time", i+1, elapsed)
	}
}
