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

	BestNetwork   Network
	BestNetworkID uint
	Matches       []Match

	Description     string
	TrainParameters string
	Active          bool
}

type Network struct {
	ID        uint `gorm:"primary_key"`
	CreatedAt time.Time

	TrainingRunID uint

	Sha  string
	Path string

	Layers  int
	Filters int
}

type Match struct {
	gorm.Model

	TrainingRunID uint

	Candidate     Network
	CandidateID   uint
	CurrentBest   Network
	CurrentBestID uint

	Wins   int
	Losses int
	Draws  int

	GameCap int
	Done    bool
}

type MatchGame struct {
	ID        uint64 `gorm:"primary_key"`
	CreatedAt time.Time

	User    User
	UserID  uint
	Match   Match
	MatchID uint

	Version uint
	Pgn     string
}

type TrainingGame struct {
	ID        uint64 `gorm:"primary_key"`
	CreatedAt time.Time

	User          User
	UserID        uint
	TrainingRun   TrainingRun
	TrainingRunID uint
	Network       Network
	NetworkID     uint

	Version   uint
	Path      string
	Pgn       string
	Compacted bool
}
