package main

import (
	"compress/gzip"
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
)

func uploadGame(c *gin.Context) {
	user := c.PostForm("user")

	// Source
	file, err := c.FormFile("file")
	if err != nil {
		c.String(http.StatusBadRequest, fmt.Sprintf("get form err: %s", err.Error()))
		return
	}

	if err := c.SaveUploadedFile(file, file.Filename); err != nil {
		c.String(http.StatusBadRequest, fmt.Sprintf("upload file err: %s", err.Error()))
		return
	}

	c.String(http.StatusOK, fmt.Sprintf("File %s uploaded successfully with fields user=%s.", file.Filename, user))
}

func main() {
	router := gin.Default()
	router.MaxMultipartMemory = 32 << 20 // 8 MiB
	router.Static("/", "./public")
	router.POST("/upload_game", uploadGame)
	router.Run(":8080")
}
