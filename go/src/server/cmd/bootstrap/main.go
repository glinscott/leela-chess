package main

import "github.com/glinscott/leela-chess/go/src/server/db"

func main() {
	db.Init(true)
	db.SetupDB()
	db.CreateTrainingRun("")
	defer db.Close()
}
