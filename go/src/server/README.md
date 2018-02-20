## Setup

Install nginx as the proxy:
```
sudo apt-get install ufw
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw enable

sudo netstat -tupln
# Ensure nothing active.
sudo apt-get purge rpcbind
sudo apt-get purge apache2 apache2-utils apache2.2-bin

# Install nginx
sudo apt-get install -y nginx
sudo systemctl status nginx

cp nginx/default /etc/nginx/sites-available/default
```

Installing postgres:
```
$ sudo apt-get install postgresql postgresql-contrib
$ sudo -u postgres createuser --interactive
Enter name of role to add: gorm
Shall the new role be a superuser? (y/n) n
Shall the new role be allowed to create databases? (y/n) y
Shall the new role be allowed to create more new roles? (y/n) n

$ sudo -u postgres createdb gorm
$ sudo -u postgres psql
ALTER ROLE gorm WITH PASSWORD 'gorm';
\q
```

### Server prereqs

```
go get github.com/gin-gonic/gin
go get -u github.com/jinzhu/gorm
go get github.com/lib/pq
go build main.go
```

In `~/.bashrc`:
```
export PATH=$PATH:/usr/lib/go-1.9/bin
export GOPATH=~/go:~/leela-chess/go
```

### Run the Server

```
./prod.sh
```

### Uploading new networks

```
curl -F 'file=@weights.txt.gz' -F 'training_id=1' -F 'layers=6' -F 'filters=64' http://localhost:8080/upload_network
```

### Server maintenance

Connecting through psql:
```
sudo -u postgres psql -d gorm
```
