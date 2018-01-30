package main

import (
	"io/ioutil"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"server/db"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"

	"client/http"
)

type StoreSuite struct {
	suite.Suite

	router *gin.Engine
	w      *httptest.ResponseRecorder
}

func (s *StoreSuite) SetupSuite() {
	db.Init(false)

	s.router = setupRouter()
}

func (s *StoreSuite) SetupTest() {
	err := db.GetDB().DropTable(
		&db.User{},
		&db.TrainingRun{},
		&db.Network{},
		&db.Match{},
		&db.MatchGame{},
		&db.TrainingGame{},
	).Error
	if err != nil {
		log.Fatal(err)
	}
	db.SetupDB()

	network := db.Network{Sha: "abcd", Path: "/tmp/network"}
	if err := db.GetDB().Create(&network).Error; err != nil {
		log.Fatal(err)
	}

	training_run := db.TrainingRun{Name: "Testing", BestNetwork: network}
	if err := db.GetDB().Create(&training_run).Error; err != nil {
		log.Fatal(err)
	}

	s.w = httptest.NewRecorder()
}

func (s *StoreSuite) TearDownSuite() {
	db.Close()
}

// This is the actual "test" as seen by Go, which runs the tests defined below
func TestStoreSuite(t *testing.T) {
	s := new(StoreSuite)
	suite.Run(t, s)
}

func (s *StoreSuite) TestNextGame() {
	req, _ := http.NewRequest("POST", "/next_game", nil)
	s.router.ServeHTTP(s.w, req)

	assert.Equal(s.T(), 200, s.w.Code, s.w.Body.String())
	assert.JSONEqf(s.T(), `{"type":"train","sha":"abcd"}`, s.w.Body.String(), "Body incorrect")
}

func (s *StoreSuite) TestUploadGameNewUser() {
	extraParams := map[string]string{
		"user":     "foo",
		"password": "asdf",
	}
	tmpfile, _ := ioutil.TempFile("", "example")
	defer os.Remove(tmpfile.Name())
	req, err := client.BuildUploadRequest("/upload_game", extraParams, "file", tmpfile.Name())
	if err != nil {
		log.Fatal(err)
	}
	s.router.ServeHTTP(s.w, req)

	assert.Equal(s.T(), 200, s.w.Code, s.w.Body.String())
}
