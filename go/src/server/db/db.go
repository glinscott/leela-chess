package db

import (
	"fmt"
	"log"

	"github.com/jinzhu/gorm"
	// Importing to support postgre database.
	_ "github.com/jinzhu/gorm/dialects/postgres"
	"server/config"
)

var db *gorm.DB
var err error

// Init initializes database.
func Init() {
	conn := fmt.Sprintf(
		"host=%s user=%s dbname=%s sslmode=disable password=%s",
		config.Config.Database.Host,
		config.Config.Database.User,
		config.Config.Database.Dbname,
		config.Config.Database.Password,
	)
	db, err = gorm.Open("postgres", conn)
	if err != nil {
		log.Fatal("Unable to connect to DB", err)
	}
}

// SetupDB setups DB.
func SetupDB() {
	db.AutoMigrate(&User{})
	db.AutoMigrate(&TrainingRun{})
	db.AutoMigrate(&Network{})
	db.AutoMigrate(&Match{})
	db.AutoMigrate(&MatchGame{})
	db.AutoMigrate(&TrainingGame{})
}

// CreateTrainingRun creates training run
func CreateTrainingRun(description string) *TrainingRun {
	trainingRun := TrainingRun{Description: description}
	err := db.Create(&trainingRun).Error
	if err != nil {
		log.Fatal(err)
	}
	return &trainingRun
}

// GetDB returns current database object
func GetDB() *gorm.DB {
	return db
}

// Close closes database
func Close() {
	db.Close()
}
