package main

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"server/db"
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

func tarGames(games []db.TrainingGame) {
	dir, err := ioutil.TempDir("", "example")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	outputTar, err := os.Create(fmt.Sprintf("games%d.tar.gz", games[0].ID))
	if err != nil {
		log.Fatalln(err)
	}
	defer outputTar.Close()
	gw := gzip.NewWriter(outputTar)
	defer gw.Close()
	tw := tar.NewWriter(gw)
	defer tw.Close()

	for _, game := range games {
		if !strings.HasSuffix(game.Path, ".gz") {
			log.Fatal("Not reading gz file?")
		}

		path := filepath.Base(game.Path)
		path = filepath.Join(dir, path[0:len(path)-3])
		log.Printf("Compressing %s to %s\n", game.Path, path)

		gzFile, err := os.Open("../../" + game.Path)
		if err != nil {
			log.Fatal(err)
		}
		defer gzFile.Close()
		gzr, err := gzip.NewReader(gzFile)
		if err != nil {
			log.Fatal(err)
		}
		defer gzr.Close()

		tmpFile, err := os.Create(path)
		if err != nil {
			log.Fatal(err)
		}
		_, err = io.Copy(tmpFile, gzr)
		if err != nil {
			log.Fatal(err)
		}
		tmpFile.Close()

		err = addFile(tw, path)
		if err != nil {
			log.Fatal(err)
		}
	}
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	db.Init(true)
	defer db.Close()

	// Query for all the active games we haven't yet compacted.
	games := []db.TrainingGame{}
	err := db.GetDB().Debug().Order("id asc").Limit(10).Where("compacted = false AND id > 17").Find(&games).Error
	if err != nil {
		log.Fatal(err)
	}

	tarGames(games)
	for _, game := range games {
		err = db.GetDB().Model(&game).Update("compacted", true).Error
		if err != nil {
			log.Fatal(err)
		}
	}
}
