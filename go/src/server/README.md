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

# Create cache directory
mkdir -p /home/web/nginx/cache/
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

$ sudo -u postgres psql -d gorm
GRANT SELECT ON ALL TABLES IN SCHEMA public TO web;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO web;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO web;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
   GRANT SELECT ON TABLES TO web;
\q
```

Setting up materialized views:
```
gorm=# CREATE MATERIALIZED VIEW games_month AS SELECT user_id, username, count(*) FROM training_games
LEFT JOIN users
ON users.id = training_games.user_id
WHERE training_games.created_at >= now() - INTERVAL '1 month'
GROUP BY user_id, username
ORDER BY count DESC;
SELECT 1606
gorm=# CREATE MATERIALIZED VIEW games_all AS SELECT user_id, username, count(*) FROM training_games
LEFT JOIN users
ON users.id = training_games.user_id
GROUP BY user_id, username
ORDER BY count DESC;
SELECT 3974
```

Then in crontab:
```
REFRESH MATERIALIZED VIEW games_month;
REFRESH MATERIALIZED VIEW games_all;
```

### Server prereqs

```
go get github.com/gin-gonic/gin
go get github.com/gin-contrib/multitemplate
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

Restarting nginx:
```
sudo service nginx restart
```

Postgres online repack
```
sudo apt-get install postgresql-server-dev-9.5 mawk
sudo easy_install pgxnclient
sudo pgxn install pg_repack
sudo -u postgres psql -c "CREATE EXTENSION pg_repack" -d gorm
/usr/lib/postgresql/9.5/bin/pg_repack
```

Postgres performance tuning
```
https://github.com/jfcoz/postgresqltuner
```

### Setting up backup

```
sudo pip install awscli

# Set up IAM user with permissions to upload to s3
aws configure
```

Executing a backup:
```
pg_dump gorm | gzip > backup.gz
```

Restoring from a backup:
```
$ dropdb -U gorm gorm
$ createdb -U gorm gorm
$ gunzip -c backup.gz | psql gorm
```

Note that on my mac, all the postgres utilities are at `/Library/PostgreSQL/10/bin/`.
