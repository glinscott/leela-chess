package main

import (
	// "compress/gzip"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"server/db"
	"strconv"

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

func uploadGame(c *gin.Context) {
	var user db.User
	user.Password = c.PostForm("password")
	err := db.GetDB().Where(db.User{Username: c.PostForm("user")}).FirstOrInit(&user).Error
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

	var training_run db.TrainingRun
	training_id, err := strconv.ParseUint(c.PostForm("training_id"), 10, 32)
	if err != nil {
		log.Println(err)
		c.String(http.StatusBadRequest, "training_id is not uint")
		return
	}
	err = db.GetDB().Where("id = ?", training_id).First(&training_run).Error
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
		TrainingRun: training_run,
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

	c.String(http.StatusOK, fmt.Sprintf("File %s uploaded successfully with fields user=%s.", file.Filename, user))
}

func setupRouter() *gin.Engine {
	router := gin.Default()
	router.MaxMultipartMemory = 32 << 20 // 8 MiB
	router.Static("/", "./public")
	router.POST("/next_game", nextGame)
	router.POST("/upload_game", uploadGame)
	return router
}

func main() {
	db.Init(true)
	db.SetupDB()
	defer db.Close()

	router := setupRouter()
	router.Run(":8080")
}
