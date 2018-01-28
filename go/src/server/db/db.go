package db

import (
	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/postgres"
	"log"
)

var db *gorm.DB
var err error

func Init() {
	db, err = gorm.Open("postgres", "host=localhost user=gorm dbname=gorm sslmode=disable password=gorm")
	if err != nil {
		log.Fatal("Unable to connect to DB", err)
	}

	db.AutoMigrate(&User{})
}

func GetDB() *gorm.DB {
	return db
}

func Close() {
	db.Close()
}
