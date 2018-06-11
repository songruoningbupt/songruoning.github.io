### 使用SourceTree和Intellij IDEA连接失败

```
error:1407742E:SSL routines:SSL23_GET_SERVER_HELLO:tlsv1 alert protocol version
```

本地环境GIT 1.9

原因：Git官方说明 https://githubengineering.com/crypto-removal-notice/

```
we’ll start disabling the following:

- TLSv1/TLSv1.1: This applies to all HTTPS connections, including web, API, and git connections to https://github.com and https://api.github.com.
- diffie-hellman-group1-sha1: This applies to all SSH connections to github.com
- diffie-hellman-group14-sha1: This applies to all SSH connections to github.com

```

解决办法：升级GIT至最新版，重新配置SourceTree和Intellij IDEA的GIT配置

[home](README.md)