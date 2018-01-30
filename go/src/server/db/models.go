package db

import (
	"time"

	"github.com/jinzhu/gorm"
)

type User struct {
	gorm.Model

	Username string
	Password string
}

type TrainingRun struct {
	gorm.Model

	BestNetwork Network
	Matches     []Match

	Name string
}

type Network struct {
	ID        uint `gorm:"primary_key"`
	CreatedAt time.Time

	TrainingRunID uint

	Sha  string
	Path string
}

type Match struct {
	gorm.Model

	TrainingRunID uint

	Candidate   Network
	CurrentBest Network

	Wins   int
	Losses int
	Draws  int

	GameCap int
	Done    bool
}

type MatchGame struct {
	ID        uint64 `gorm:"primary_key"`
	CreatedAt time.Time

	User  User
	Match Match

	Version uint
	Pgn     string
}

type TrainingGame struct {
	ID        uint64 `gorm:"primary_key"`
	CreatedAt time.Time

	User        User
	TrainingRun TrainingRun
	Network     Network

	Version   uint
	Path      string
	Pgn       string
	Compacted bool
}
