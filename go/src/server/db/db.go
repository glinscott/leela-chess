package db

import (
	"fmt"
	"log"

	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/postgres"
)

var db *gorm.DB
var err error

func Init(prod bool) {
	dbname := "gorm_test"
	if prod {
		dbname = "gorm"
	}
	conn := fmt.Sprintf("host=localhost user=gorm dbname=%s sslmode=disable password=gorm", dbname)
	db, err = gorm.Open("postgres", conn)
	if err != nil {
		log.Fatal("Unable to connect to DB", err)
	}
}

func SetupDB() {
	db.AutoMigrate(&User{})
	db.AutoMigrate(&TrainingRun{})
	db.AutoMigrate(&Network{})
	db.AutoMigrate(&Match{})
	db.AutoMigrate(&MatchGame{})
	db.AutoMigrate(&TrainingGame{})
}

func CreateTrainingRun(description string) *TrainingRun {
	training_run := TrainingRun{Description: description}
	err := db.Create(&training_run).Error
	if err != nil {
		log.Fatal(err)
	}
	return &training_run
}

func GetDB() *gorm.DB {
	return db
}

func Close() {
	db.Close()
}
