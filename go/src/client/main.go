package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"time"

	"client/http"
)

var HOSTNAME = flag.String("hostname", "http://162.217.248.187", "Address of the server")
var USER = flag.String("user", "", "Username")
var PASSWORD = flag.String("password", "", "Password")
var GPU = flag.Int("gpu", 0, "ID of the OpenCL device to use")

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

func uploadGame(httpClient *http.Client, path string, train_dir string, pgn string, nextGame client.NextGameResponse) error {
	extraParams := map[string]string{
		"user":        *USER,
		"password":    *PASSWORD,
		"version":     "1",
		"training_id": strconv.Itoa(int(nextGame.TrainingId)),
		"network_id":  strconv.Itoa(int(nextGame.NetworkId)),
		"pgn":         pgn,
	}
	request, err := client.BuildUploadRequest(*HOSTNAME+"/upload_game", extraParams, "file", path)
	if err != nil {
		return err
	}
	resp, err := httpClient.Do(request)
	if err != nil {
		return err
	}
	body := &bytes.Buffer{}
	_, err = body.ReadFrom(resp.Body)
	if err != nil {
		return err
	}
	resp.Body.Close()
	fmt.Println(resp.StatusCode)
	fmt.Println(resp.Header)
	fmt.Println(body)

	// Delete file and dir, as they are no longer used.
	err = os.RemoveAll(train_dir)
	if err != nil {
		log.Fatal(err)
	}

	return nil
}

/*
func playMatch() {
	p1 := exec.Command("lczero")
  p1_in, _ := p1.StdinPipe()
  p1_out, _ := p1.StdoutPipe()
  p1.Start()
  p1.Write("...")
}
*/

func train(networkPath string) (string, string, string) {

	// Try to make a unique id by combining random, time and pid. (used for multithreading)
	pid := os.Getpid()
	rand_number := rand.Intn(10000000)
	time_now := time.Now().Unix()

	unique_id := fmt.Sprintf("%v-%v-%v", pid, time_now, rand_number)
	dir_name := fmt.Sprintf("data-%v", unique_id)

	dir, _ := os.Getwd()
	train_dir := path.Join(dir, dir_name)
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

	num_games := 1
	gpu_id := fmt.Sprintf("--gpu=%v", *GPU)
	train_cmd := fmt.Sprintf("--start=train %v %v", unique_id, num_games)
	weights := fmt.Sprintf("--weights=%s", networkPath)

	// cmd := exec.Command(path.Join(dir, "lczero"), weights, "--randomize", "-n", "-t1", "-p20", "--noponder", "--quiet", train_cmd)
	cmd := exec.Command(path.Join(dir, "lczero"), weights, gpu_id, "--randomize", "-n", "-t1", "--quiet", train_cmd)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
	}
	stdoutScanner := bufio.NewScanner(stdout)
	pgn := ""
	go func() {
		reading_pgn := false
		for stdoutScanner.Scan() {
			line := stdoutScanner.Text()
			fmt.Printf("%s\n", line)
			if line == "PGN" {
				reading_pgn = true
			} else if line == "END" {
				reading_pgn = false
			} else if reading_pgn {
				pgn += line + "\n"
			}
		}
	}()

	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Fatal(err)
	}
	stderrScanner := bufio.NewScanner(stderr)
	go func() {
		for stderrScanner.Scan() {
			fmt.Printf("%s\n", stderrScanner.Text())
		}
	}()

	err = cmd.Start()
	if err != nil {
		log.Fatal(err)
	}

	err = cmd.Wait()
	if err != nil {
		log.Fatal(err)
	}

	return path.Join(train_dir, "training.0.gz"), pgn, train_dir
}

func getNetwork(httpClient *http.Client, sha string) (string, error) {
	// Sha already exists?
	path := filepath.Join("networks", sha)
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}

	// Clean out any old networks
	os.RemoveAll("networks")
	os.MkdirAll("networks", os.ModePerm)

	fmt.Printf("Downloading network...\n")
	// Otherwise, let's download it
	err := client.DownloadNetwork(httpClient, *HOSTNAME, path, sha)
	if err != nil {
		return "", err
	}
	return path, nil
}

func nextGame(httpClient *http.Client, hostname string) error {
	nextGame, err := client.NextGame(httpClient, *HOSTNAME)
	if err != nil {
		return err
	}
	networkPath, err := getNetwork(httpClient, nextGame.Sha)
	if err != nil {
		return err
	}
	trainFile, pgn, train_dir := train(networkPath)

	go uploadGame(httpClient, trainFile, train_dir, pgn, nextGame)
	return nil
}

func main() {
	flag.Parse()

	if len(*USER) == 0 || len(*PASSWORD) == 0 {
		*USER, *PASSWORD = readSettings("settings.json")
	}

	if len(*USER) == 0 {
		log.Fatal("You must specify a username")
	}
	if len(*PASSWORD) == 0 {
		log.Fatal("You must specify a non-empty password")
	}

	httpClient := &http.Client{}
	for {
		err := nextGame(httpClient, *HOSTNAME)
		if err != nil {
			log.Print(err)
			log.Print("Sleeping for 30 seconds...")
			time.Sleep(30 * time.Second)
			continue
		}
	}
}
