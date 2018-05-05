# Compiling

You will need to install Go 1.9 or later.

Then, make sure to set up your GOPATH properly, eg. here is mine:
```
export GOPATH=/Users/gary/go:/Users/gary/Development/leela-chess/go
```
Here, I've set my system install of go as the first entry, and then the leela-chess/go directory as the second.

Pre-reqs:
```
# (Bug workaround, using Tilps instead)
# go get -u github.com/notnil/chess
go get -u github.com/Tilps/chess

```

Then you just need to `go build`, and it should produce a `client` executable.

# Running

First copy the `lczero` executable into the same folder as the `client` executable.

Then, run!  Username and password are required parameters.
```
./client --user=myusername --password=mypassword
```

For testing, you can also point the client at a different server:
```
./client --hostname=http://127.0.0.1:8080 --user=test --password=asdf
```

# Cross-compiling

One of the main reasons I picked go was it's amazing support for cross-compiling.

Pre-reqs:
```
GOOS=windows GOARCH=amd64 go install
GOOS=darwin GOARCH=amd64 go install
GOOS=linux GOARCH=amd64 go install
```

Building the client for each platform:
```
GOOS=windows GOARCH=amd64 go build -o client.exe
GOOS=darwin GOARCH=amd64 go build -o client_mac
GOOS=linux GOARCH=amd64 go build -o client_linux
```
