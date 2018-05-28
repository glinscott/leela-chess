package main

import (
	"log"
	"server/db"
)

func updateNetworkCounts() {
	rows, err := db.GetDB().Raw(`SELECT network_id, count(*) FROM training_games GROUP BY network_id`).Rows()
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	for rows.Next() {
		var network_id uint
		var count uint64
		rows.Scan(&network_id, &count)
		err := db.GetDB().Exec("UPDATE networks SET games_played=? WHERE id=?", count, network_id).Error
		if err != nil {
			log.Fatal(err)
		}
	}
}

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
	//training_run.Active = true
	//training_run.Description = "Initial testing run"
	training_run.TrainParameters = `["--randomize", "-n", "-v800"]`
	err = db.GetDB().Save(&training_run).Error
	if err != nil {
		log.Fatal(err)
	}
}

func newMatch() {
	match := db.Match{
		TrainingRunID: 1,
		CandidateID:   237,
		CurrentBestID: 316,
		Done:          false,
		GameCap:       400,
		Parameters:    `["--tempdecay=10"]`,
		TestOnly:      true,
	}
	err := db.GetDB().Create(&match).Error
	if err != nil {
		log.Fatal(err)
	}
}

func setTestOnly() {
	match := db.Match{}
	match.ID = 90
	err := db.GetDB().Where(&match).First(&match).Error
	if err != nil {
		log.Fatal(err)
	}
	match.TestOnly = true
	err = db.GetDB().Save(&match).Error
	if err != nil {
		log.Fatal(err)
	}
}

func updateMatchPassed() {
	var matches []db.Match
	err := db.GetDB().Find(&matches).Error
	if err != nil {
		log.Fatal(err)
	}
	for _, match := range matches {
		match.Passed = match.Wins > match.Losses
		err = db.GetDB().Save(&match).Error
		if err != nil {
			log.Fatal(err)
		}
	}
}

/*
func dumpPgns() {
	start := 9168243
	end := 10014931
	for id := start; id < end; id++ {
		fmt.Printf("\r%d",id)
		game := db.TrainingGame{
			ID: uint64(id),
		}
		err := db.GetDB().Where(&game).First(&game).Error
		if err != nil {
			fmt.Printf(" - skipping\n")
			continue
		}
		pgn_path := fmt.Sprintf("/home/web/leela-chess/go/src/server/pgns/run1/%d.pgn", game.ID)
		err = ioutil.WriteFile(pgn_path, []byte(game.Pgn), 0644)
		if err != nil {
			log.Fatal(err)
		}
	}
	fmt.Printf("\n")
}
*/

func main() {
	db.Init(true)
	db.SetupDB()

	// newRun()
	// makeRunActive()
	// newMatch()
	// setTestOnly()
	// updateNetworkCounts()
	// updateMatchPassed()
	// dumpPgns()

	defer db.Close()
}
