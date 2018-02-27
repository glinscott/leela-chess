package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"io"
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
	assert.JSONEqf(s.T(), `{"type":"train","trainingId":1,"networkId":1,"sha":"abcd"}`, s.w.Body.String(), "Body incorrect")
}

func (s *StoreSuite) TestUploadGameNewUser() {
	extraParams := map[string]string{
		"user":        "foo",
		"password":    "asdf",
		"training_id": "1",
		"network_id":  "1",
		"version":     "1",
	}
	tmpfile, _ := ioutil.TempFile("", "example")
	defer os.Remove(tmpfile.Name())
	req, err := client.BuildUploadRequest("/upload_game", extraParams, "file", tmpfile.Name())
	if err != nil {
		log.Fatal(err)
	}
	s.router.ServeHTTP(s.w, req)

	assert.Equal(s.T(), 200, s.w.Code, s.w.Body.String())

	user := db.User{}
	err = db.GetDB().Where("username = ?", "foo").First(&user).Error
	if err != nil {
		log.Fatal(err)
	}
}

func uploadTestNetwork(s *StoreSuite, contentString string, networkId int) {
	s.w = httptest.NewRecorder()
	content := []byte(contentString)
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	zw.Write(content)
	zw.Close()

	extraParams := map[string]string{
		"training_id": "1",
		"layers":      "6",
		"filters":     "64",
	}
	tmpfile, _ := ioutil.TempFile("", "example")
	defer os.Remove(tmpfile.Name())
	if _, err := tmpfile.Write(buf.Bytes()); err != nil {
		log.Fatal(err)
	}
	req, err := client.BuildUploadRequest("/upload_network", extraParams, "file", tmpfile.Name())
	if err != nil {
		log.Fatal(err)
	}
	s.router.ServeHTTP(s.w, req)

	assert.Equal(s.T(), 200, s.w.Code, s.w.Body.String())

	// Trying to upload the same network should now fail
	s.w = httptest.NewRecorder()
	req, err = client.BuildUploadRequest("/upload_network", extraParams, "file", tmpfile.Name())
	if err != nil {
		log.Fatal(err)
	}
	s.router.ServeHTTP(s.w, req)
	assert.Equal(s.T(), 400, s.w.Code, s.w.Body.String())

	// Now we should be able to query for this network
	s.w = httptest.NewRecorder()
	sha := sha256.Sum256(content)
	req, _ = http.NewRequest("POST", "/next_game", nil)
	s.router.ServeHTTP(s.w, req)
	assert.Equal(s.T(), 200, s.w.Code, s.w.Body.String())
	assert.JSONEqf(s.T(), fmt.Sprintf(`{"type":"train","trainingId":1,"networkId":%d,"sha":"%x"}`, networkId, sha), s.w.Body.String(), "Body incorrect")

	// And let's download it now.
	s.w = httptest.NewRecorder()
	req, _ = http.NewRequest("GET", fmt.Sprintf("/get_network?sha=%x", sha), nil)
	s.router.ServeHTTP(s.w, req)
	assert.Equal(s.T(), 200, s.w.Code, s.w.Body.String())

	// Should match the contents
	zr, err := gzip.NewReader(s.w.Body)
	if err != nil {
		log.Fatal(err)
	}
	buf.Reset()
	if _, err := io.Copy(&buf, zr); err != nil {
		log.Fatal(err)
	}
	assert.Equal(s.T(), contentString, buf.String(), "Contents don't match")
}

func (s *StoreSuite) TestUploadNetwork() {
	uploadTestNetwork(s, "this_is_a_network", 2)
	uploadTestNetwork(s, "network2", 3)
}
