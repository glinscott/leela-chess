## Setup

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

### Server prereqs

```
go get github.com/gin-gonic/gin
go get -u github.com/jinzhu/gorm
go get github.com/lib/pq
go build main.go
```

### Run the Server

```
./prod.sh
```
