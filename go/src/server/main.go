package main

import (
	// "compress/gzip"
	"fmt"
	"net/http"
	"path/filepath"
	"server/db"
	"strconv"

	"github.com/gin-gonic/gin"
)

func uploadGame(c *gin.Context) {
	var user db.User
	gerr := db.GetDB().Where(db.User{Username: c.PostForm("user")}).FirstOrInit(&user)
	if gerr != nil {
		c.String(http.StatusBadRequest, fmt.Sprintf("Invalid user: %s", gerr))
		return
	}

	// Ensure passwords match
	if user.Password != c.PostForm("password") {
		c.String(http.StatusBadRequest, fmt.Sprintf("Incorrect password"))
		return
	}

	var training_run db.TrainingRun
	gerr = db.GetDB().Where("id = ?", c.PostForm("training_id")).First(&training_run)
	if gerr != nil {
		c.String(http.StatusBadRequest, fmt.Sprintf("Invalid training run: %s", gerr))
		return
	}

	var network db.Network
	gerr = db.GetDB().Where("id = ?", c.PostForm("network_id")).First(&network)
	if gerr != nil {
		c.String(http.StatusBadRequest, fmt.Sprintf("Invalid network: %s", gerr))
		return
	}

	// Source
	file, err := c.FormFile("file")
	if err != nil {
		c.String(http.StatusBadRequest, fmt.Sprintf("get form err: %s", err.Error()))
		return
	}

	// Create new game
	version, err := strconv.ParseUint(c.PostForm("version"), 10, 64)
	if err != nil {
		c.String(http.StatusBadRequest, fmt.Sprintf("Invalid version: %s", err.Error()))
	}
	game := db.Game{UserID: user.ID, TrainingRunID: training_run.ID, NetworkID: network.ID, Version: uint(version), Pgn: c.PostForm("pgn")}
	db.GetDB().Create(&game)
	db.GetDB().Model(&game).Update("path", filepath.Join("games", fmt.Sprintf("run%d/training.%d.gz", training_run.ID, game.ID)))

	// Save in the directory
	if err := c.SaveUploadedFile(file, game.Path); err != nil {
		c.String(http.StatusBadRequest, fmt.Sprintf("upload file err: %s", err.Error()))
		return
	}

	c.String(http.StatusOK, fmt.Sprintf("File %s uploaded successfully with fields user=%s.", file.Filename, user))
}

func main() {
	db.Init()
	defer db.Close()

	router := gin.Default()
	router.MaxMultipartMemory = 32 << 20 // 8 MiB
	router.Static("/", "./public")
	router.POST("/upload_game", uploadGame)
	router.Run(":8080")
}
