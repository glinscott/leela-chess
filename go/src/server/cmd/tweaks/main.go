package main

import (
	"log"
	"server/db"
)

func newRun() {
	training_run := db.CreateTrainingRun("v0.2 6x64 Random start")
	training_run.Active = true
	training_run.TrainParameters = `["--randomize", "-n"]`
	err := db.GetDB().Save(&training_run).Error
	if err != nil {
		log.Fatal(err)
	}
}

func makeRunActive() {
	training_run := db.TrainingRun{}
	err := db.GetDB().Where(&training_run).First(&training_run).Error
	if err != nil {
		log.Fatal(err)
	}
	training_run.Active = true
	training_run.Description = "Initial testing run"
	training_run.TrainParameters = `["--randomize", "-n"]`
	err = db.GetDB().Save(&training_run).Error
	if err != nil {
		log.Fatal(err)
	}
}

func newMatch() {
	match := db.Match{
		TrainingRunID: 1,
		CandidateID:   4,
		CurrentBestID: 3,
		Wins:          89,
		Losses:        4,
		Draws:         7,
		Done:          true,
	}
	err := db.GetDB().Create(&match).Error
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	db.Init(true)
	db.SetupDB()

	// newRun()
	// makeRunActive()
	newMatch()

	defer db.Close()
}
