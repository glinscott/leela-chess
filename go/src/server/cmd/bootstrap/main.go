package main

import (
	"server/db"
)

func main() {
	db.Init(true)
	db.SetupDB()
	db.CreateTrainingRun()
	defer db.Close()
}
