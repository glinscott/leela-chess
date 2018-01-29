package main

import (
	"net/http"
	"net/http/httptest"
	"server/db"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
)

type StoreSuite struct {
	suite.Suite

	router *gin.Engine
	w      *httptest.ResponseRecorder
}

func (s *StoreSuite) SetupSuite() {
	db.Init(false)

	s.router = setupRouter()
	s.w = httptest.NewRecorder()
}

func (s *StoreSuite) SetupTest() {
	//db.DropTable(
	db.SetupDB()
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
}
