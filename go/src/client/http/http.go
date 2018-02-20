package client

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
)

func postJson(httpClient *http.Client, uri string, target interface{}) error {
	r, err := httpClient.Post(uri, "application/json", bytes.NewBuffer([]byte{}))
	if err != nil {
		return err
	}
	defer r.Body.Close()
	b, _ := ioutil.ReadAll(r.Body)
	err = json.Unmarshal(b, target)
	if err != nil {
		log.Printf("Bad JSON from %s -- %s\n", uri, string(b))
	}
	return err
}

// Creates a new file upload http request with optional extra params
func BuildUploadRequest(uri string, params map[string]string, paramName, path string) (*http.Request, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile(paramName, filepath.Base(path))
	if err != nil {
		return nil, err
	}
	_, err = io.Copy(part, file)

	for key, val := range params {
		_ = writer.WriteField(key, val)
	}
	err = writer.Close()
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", uri, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	return req, err
}

type NextGameResponse struct {
	Type       string
	TrainingId uint
	NetworkId  uint
	Sha        string
}

func NextGame(httpClient *http.Client, hostname string) (NextGameResponse, error) {
	resp := NextGameResponse{}
	err := postJson(httpClient, hostname+"/next_game", &resp)

	if len(resp.Sha) == 0 {
		return resp, errors.New("Server gave back empty SHA")
	}

	return resp, err
}

func DownloadNetwork(httpClient *http.Client, hostname string, networkPath string, sha string) error {
	uri := hostname + fmt.Sprintf("/get_network?sha=%s", sha)
	r, err := httpClient.Get(uri)
	defer r.Body.Close()
	if err != nil {
		return err
	}

	out, err := os.Create(networkPath)
	defer out.Close()
	if err != nil {
		return err
	}

	// Copy while decompressing
	zr, err := gzip.NewReader(r.Body)
	if err != nil {
		log.Fatal(err)
	}

	_, err = io.Copy(out, zr)
	return err
}
