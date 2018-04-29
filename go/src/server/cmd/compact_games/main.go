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
	"server/db"
	"sort"
	"strconv"
	"strings"
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

func tarGame(game *db.TrainingGame, dir string, tw *tar.Writer) error {
	if len(game.Path) == 0 {
		log.Printf("Skipping empty path\n")
		return nil
	}

	if !strings.HasSuffix(game.Path, ".gz") {
		log.Fatal("Not reading gz file?")
	}

	path := filepath.Base(game.Path)
	path = filepath.Join(dir, path[0:len(path)-3])
	// log.Printf("Compressing %s to %s\n", game.Path, path)

	gzFile, err := os.Open("../../" + game.Path)
	if err != nil {
		log.Fatal(err)
	}
	defer gzFile.Close()
	gzr, err := gzip.NewReader(gzFile)
	if err != nil {
		log.Printf("Skipping %s: %v\n", path, err)
		return nil
	}
	defer gzr.Close()

	tmpFile, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer tmpFile.Close()
	_, err = io.Copy(tmpFile, gzr)
	if err != nil {
		return err
	}

	err = addFile(tw, path)
	if err != nil {
		log.Fatal(err)
	}

	// Remove the temporary file
	err = os.Remove(path)
	if err != nil {
		log.Fatal(err)
	}

	return nil
}

func tarGames(games []db.TrainingGame) string {
	dir, err := ioutil.TempDir("", "example")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	outputPath := fmt.Sprintf("games%d.tar.gz", games[0].ID / 10000 * 10000)
	outputTar, err := os.Create(outputPath)
	if err != nil {
		log.Fatalln(err)
	}
	defer outputTar.Close()
	gw := gzip.NewWriter(outputTar)
	defer gw.Close()
	tw := tar.NewWriter(gw)
	defer tw.Close()

	fmt.Printf("Starting at game %d\n", games[0].ID)
	for idx, game := range games {
		fmt.Printf("\r%d/%d games", idx, len(games))

		err = tarGame(&game, dir, tw)
		if err != nil {
			fmt.Println()
			log.Print(err)
		}
	}
	fmt.Println()

	return outputPath
}

func deleteCompactedGames() {
	dir := "../../games/run1/"
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		log.Fatal(err)
	}

	ids := []int{}
	for _, file := range files {
		id, err := strconv.Atoi(strings.Split(file.Name(), ".")[1])
		if err != nil {
			log.Fatal(err)
		}
		ids = append(ids, id)
	}
	sort.Ints(ids)

	// Leave this many games on the server
	leaveGames := 500000
	log.Printf("Deleting from %d\n", ids[0])
	for _, id := range ids {
		if id + leaveGames >= ids[len(ids)-1] {
			log.Printf("Deleted to %d\n", id)
			break
		}
	}
	log.Printf("Latest id %d\n", ids[len(ids)-1])

	for _, id := range ids {
		if id + leaveGames >= ids[len(ids)-1] {
			break
		}
		err := os.Remove(dir + "training." + strconv.Itoa(id) + ".gz")
		if err != nil {
			log.Fatal(err)
		}
	}
}

func compactGames() bool {
	// Query for all the active games we haven't yet compacted.
	games := []db.TrainingGame{}
	var numGames int64 = 10000
	err := db.GetDB().Order("id asc").Limit(numGames).Where("compacted = false AND id >= 40000").Find(&games).Error
	if err != nil {
		log.Fatal(err)
	}
	if len(games) != int(numGames) {
		return false
	}
	stop := int64(games[0].ID) / numGames * numGames + numGames
	for idx, game := range games {
		if int64(game.ID) >= stop {
			games = games[0:idx]
			break
		}
	}

	outputPath := tarGames(games)
	cmd := exec.Command("aws", "s3", "cp", outputPath, "s3://lczero/training/")
	cmd.Stdout = os.Stdout
	err = cmd.Run()
	if err != nil {
		log.Fatal(err)
	}
	err = os.Remove(outputPath)
	if err != nil {
		log.Fatal(err)
	}

	for _, game := range games {
		err = db.GetDB().Model(&game).Update("compacted", true).Error
		if err != nil {
			log.Fatal(err)
		}
	}
	return true
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	db.Init(true)
	defer db.Close()

	for compactGames() {
	}

	deleteCompactedGames()
}
