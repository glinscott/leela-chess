package main

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/marcsauter/single"
)

func addFile(tw *tar.Writer, path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	if stat, err := file.Stat(); err == nil {
		// now lets create the header as needed for this file within the tarball
		header := new(tar.Header)
		header.Name = filepath.Base(path)
		header.Size = stat.Size()
		header.Mode = int64(stat.Mode())
		header.ModTime = stat.ModTime()
		// write the header to the tarball archive
		if err := tw.WriteHeader(header); err != nil {
			return err
		}
		// copy the file data to the tarball
		if _, err := io.Copy(tw, file); err != nil {
			return err
		}
	}
	return nil
}

func upload(outputPath string) {
	cmd := exec.Command("aws", "s3", "cp", outputPath, "s3://lczero/training/run1/")
	cmd.Stdout = os.Stdout
	err := cmd.Run()
	if err != nil {
		log.Fatal(err)
	}
	err = os.Remove(outputPath)
	if err != nil {
		log.Fatal(err)
	}
}

func tarGames(dir string, games []int, startId int) string {
	outputPath := fmt.Sprintf("pgn%d.tar.gz", startId)
	outputTar, err := os.Create(outputPath)
	if err != nil {
		log.Fatalln(err)
	}
	defer outputTar.Close()
	gw := gzip.NewWriter(outputTar)
	defer gw.Close()
	tw := tar.NewWriter(gw)
	defer tw.Close()

	fmt.Printf("Starting at game %d\n", games[0])
	for idx, game := range games {
		if idx % 100 == 0 {
			fmt.Printf("\r%d/%d games", idx, len(games))
		}

		path := dir + strconv.Itoa(game) + ".pgn"
		err = addFile(tw, path)
		if err != nil {
			log.Fatal(err)
		}
	}
	fmt.Println()
	return outputPath
}

func uploadAndDelete(dir string, games []int, outputPath string) {
	log.Println("Uploading")
	upload(outputPath)

	// Delete games
	log.Println("Deleting")
	for _, game := range games {
		err := os.Remove(dir + strconv.Itoa(game) + ".pgn")
		if err != nil {
			log.Fatal(err)
		}
	}
}

func listFiles(dir string) []int {
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		log.Fatal(err)
	}

	ids := []int{}
	for _, file := range files {
		id, err := strconv.Atoi(strings.Split(file.Name(), ".")[0])
		if err != nil {
			log.Fatal(err)
		}
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	s := single.New("compact_pgns")
	if err := s.CheckLock(); err != nil && err == single.ErrAlreadyRunning {
		log.Fatal("another instance of the app is already running, exiting")
	} else if err != nil {
		// Another error occurred, might be worth handling it as well
		log.Fatalf("failed to acquire exclusive app lock: %v", err)
	}
	defer s.TryUnlock()

	dir := "../../pgns/run1/"
	ids := listFiles(dir)

	leaveGames := 500000
	chunkSize := 100000
	log.Printf("Deleting from %d (last %d)\n", ids[0], ids[len(ids)-1])
	for idx, id := range ids {
		if id + leaveGames >= ids[len(ids)-1] / chunkSize * chunkSize {
			log.Printf("Deleted to %d\n", id)
			ids = ids[0:idx]
			break
		}
	}

	if len(ids) == 0 {
		log.Println("Nothing to do")
		return
	}

	log.Printf("Latest id %d\n", ids[len(ids)-1])

	idx := 0
	for idx < len(ids) {
		startId := ids[idx] / chunkSize * chunkSize
		delta := ids[idx] - startId
		endIdx := idx+chunkSize-delta
		outputPath := tarGames(dir, ids[idx:endIdx], startId)
		uploadAndDelete(dir, ids[idx:endIdx], outputPath)
		idx = endIdx
	}
}
