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
	"path"
	"strconv"
	"time"

	"client/http"
)

var HOSTNAME = flag.String("hostname", "http://162.217.248.187/", "Address of the server")
var USER = flag.String("user", "", "Username")
var PASSWORD = flag.String("password", "", "Password")

func uploadFile(httpClient *http.Client, path string, nextGame client.NextGameResponse) {
	extraParams := map[string]string{
		"user":        *USER,
		"password":    *PASSWORD,
		"version":     "1",
		"training_id": strconv.Itoa(int(nextGame.TrainingId)),
		"network_id":  strconv.Itoa(int(nextGame.NetworkId)),
	}
	request, err := client.BuildUploadRequest(*HOSTNAME+"/upload_game", extraParams, "file", path)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := httpClient.Do(request)
	if err != nil {
		log.Fatal(err)
	}
	body := &bytes.Buffer{}
	_, err = body.ReadFrom(resp.Body)
	if err != nil {
		log.Fatal(err)
	}
	resp.Body.Close()
	fmt.Println(resp.StatusCode)
	fmt.Println(resp.Header)
	fmt.Println(body)
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

func train() string {
	pid := 1

	dir, _ := os.Getwd()
	train_dir := path.Join(dir, fmt.Sprintf("data-%v", pid))
	if _, err := os.Stat(train_dir); err == nil {
		err = os.RemoveAll(train_dir)
		if err != nil {
			log.Fatal(err)
		}
	}

	num_games := 1
	train_cmd := fmt.Sprintf("--start=train %v %v", pid, num_games)
	// cmd := exec.Command(path.Join(dir, "lczero"), "--weights=weights.txt", "--randomize", "-n", "-t1", "-p20", "--noponder", train_cmd)
	cmd := exec.Command(path.Join(dir, "lczero"), "--weights=weights.txt", "--randomize", "-n", "-t1", train_cmd)

	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Fatal(err)
	}
	scanner := bufio.NewScanner(stderr)
	go func() {
		for scanner.Scan() {
			fmt.Printf("%s\n", scanner.Text())
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

	return path.Join(train_dir, "training.0.gz")
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
		nextGame, err := client.NextGame(httpClient, *HOSTNAME)
		if err != nil {
			log.Print(err)
			log.Print("Sleeping for 30 seconds")
			time.Sleep(30 * time.Second)
		}
		trainFile := train()
		uploadFile(httpClient, trainFile, nextGame)
	}
}
