package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"io/ioutil"
	"path"
	"path/filepath"
	"strconv"
	"time"

	"client/http"
)

var HOSTNAME = flag.String("hostname", "http://162.217.248.187", "Address of the server")
var USER = flag.String("user", "", "Username")
var PASSWORD = flag.String("password", "", "Password")

func uploadGame(httpClient *http.Client, path string, pgn string, nextGame client.NextGameResponse) error {
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

func train(networkPath string) (string, string) {
	// pid is intended for use in multi-threaded training
	pid := 1

	dir, _ := os.Getwd()
	train_dir := path.Join(dir, fmt.Sprintf("data-%v", pid))
	if _, err := os.Stat(train_dir); err == nil {
		files, err := ioutil.ReadDir(train_dir)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Cleanup training files:\n");
		for _, f := range files {
			fmt.Printf("%s/%s\n", train_dir, f.Name());
		}
		err = os.RemoveAll(train_dir)
		if err != nil {
			log.Fatal(err)
		}
	}

	num_games := 1
	train_cmd := fmt.Sprintf("--start=train %v %v", pid, num_games)
	weights := fmt.Sprintf("--weights=%s", networkPath)
	// cmd := exec.Command(path.Join(dir, "lczero"), weights, "--randomize", "-n", "-t1", "-p20", "--noponder", "--quiet", train_cmd)
	cmd := exec.Command(path.Join(dir, "lczero"), weights, "--randomize", "-n", "-t1", "--quiet", train_cmd)

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

	return path.Join(train_dir, "training.0.gz"), pgn
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
	trainFile, pgn := train(networkPath)
	uploadGame(httpClient, trainFile, pgn, nextGame)
	return nil
}

func main() {
	flag.Parse()
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
