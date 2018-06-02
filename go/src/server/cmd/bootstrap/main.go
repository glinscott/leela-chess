package main

import (
	"server/db"
)

func main() {
	db.Init()
	defer db.Close()
	db.SetupDB()
	db.CreateTrainingRun("Initial run just for test")
}
