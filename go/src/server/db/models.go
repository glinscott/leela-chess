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

	Name string
}

type Network struct {
	ID        uint `gorm:"primary_key"`
	CreatedAt time.Time

	Sha string
}

type Game struct {
	ID        uint64 `gorm:"primary_key"`
	CreatedAt time.Time

	UserID        uint
	TrainingRunID uint
	NetworkID     uint

	Version uint
	Path    string
	Pgn     string
}
