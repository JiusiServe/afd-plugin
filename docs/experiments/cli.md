# 安装
## 安装 modctl
### 蚂蚁内部（蚂蚁物理机环境/使用蚂蚁内部镜像的容器）
建议通过蚂蚁内部源安装。首次安装可能需要添加蚂蚁内部源，方法如下（更详细的信息可以参考[蚂蚁制品库使用说明](https://yuque.antfin.com/antbuild/bpxc9y/aww0ho#ysZtk)）：

```shell
# 对于 7u 的操作系统：
$ sudo yum install -y http://artifacts.antgroup-inc.cn/artifact/repositories/ant_7_noarch_current/ant-repo-utils/ant-repo-utils-0.0.1-43313729.noarch.rpm

# 对于 8u 的操作系统：
$ sudo yum install -y https://artifacts.antgroup-inc.cn/artifact/repositories/ant_8_noarch_current/ant-repo-utils/ant-repo-utils-0.0.1-202188063.noarch.rpm

```

安装最新版 modctl ：

```shell
$ sudo yum install -y modctl -b current
```

### 蚂蚁内部使用 Debian/Ubuntu 等外部镜像的容器
对于Debian/Ubuntu系统，我们也上传了deb包内部蚂蚁制品库的，方便用户下载安装。因为没有蚂蚁内部 apt源，所以用户需wget下载deb包后手动安装：可访问[https://artifacts-web.antgroup-inc.cn/common/versions?name=modctl-0&t=MAIN_SITE](https://artifacts-web.antgroup-inc.cn/common/versions?name=modctl-0&t=MAIN_SITE) 找到满足自己系统架构的版本下载后通过`dpkg -i xxx.deb`来进行安装。建议尽可能选择最新版本。

### 蚂蚁外部，或者 MacOS 等办公网环境
可以选择从 github 下载安装：[modctl releases](https://github.com/modelpack/modctl/tags) 。（点击页面最新的 tag，然后下载不同平台的安装包，建议尽可能选择最新版本）

### 暂不支持 windows
# 使用
## 账号
请通过 [https://hcs.alipay.com/hmr/credentials](https://hcs.alipay.com/hmr/credentials) 设置登录密码，使用 **<font style="color:#DF2A3F;">域账号名 + HMR 密码（非域账号密码）</font>**登录（参考文档 [从命令行拉取和推送镜像](https://yuque.antfin.com/tuna/ifau4b/ovkir7124ks7xmax)）。其他问题请咨询 [@楚贤](https://yuque.antfin.com/chuxian.mjj)[@康德](https://yuque.antfin.com/lb203159)。



<img src="https://intranetproxy.alipay.com/skylark/lark/0/2025/png/199187/1766642693466-3b14f2d7-e6e2-4f5f-895f-cc283e7ae356.png" width="577" title="" crop="0,0,1,1" id="ov80r" class="ne-image">

## 登录
现有使用环境下，网络环境纷繁复杂，涉及主站集群、AIDC 集群、公有云网络、办公网环境、容器环境、宿主机环境等等，同时又要保证高性能，因此以下域名选择和命令选项非常关键，需要精心选择。

### 域名选择
办公网域名(网速不可控): `hmr.antgroup-inc.cn`

线上域名: `hmr.sa128.alipay.com`，我们建议在<font style="background-color:#FBDE28;">线上</font>环境尽可能的使用这个域名，因为前者走 spanner 转发流量受网络集群限制，本身速率不稳定。

### 命令选项
#### 关于--plain-http
如果是`hmr.antgroup-inc.cn`域名，<font style="background-color:#FBDE28;">必须要用 https，所以不加</font>`<font style="background-color:#FBDE28;">--plain-http</font>`。

如果是`hmr.sa128.alipay.com`域名，<font style="background-color:#FBDE28;">只支持 http，所以必须加</font>`<font style="background-color:#FBDE28;">--plain-http</font>`。

#### 关于--insecure
一般情况下不用考虑这个选项。只有在使用`hmr.antgroup-inc.cn`域名，并且使用`--proxy `走 dragonfly 加速时，需要加上`--insecure` 选项。

### 登录示例
例如在主站生产环境，为了尽可能高的性能所以使用域名 `hmr.sa128.alipay.com`，而 `hmr.sa128.alipay.com`目前支持 http 协议，所以需要加上`--plain-http`，命令如下：

```shell
$ modctl login hmr.sa128.alipay.com --plain-http -u ${username} -p ${password}
Logging In...
Login Succeeded.
```

## 上传
### 构建 & 上传模型
`/path/to/model`即为本地需要上传的模型目录, `${namespace}`为申请的 Namespace。

```shell
$ modctl modelfile generate /path/to/model --output /path/to/modelfile
$ modctl build -f /path/to/modelfile/Modelfile /path/to/model \
-t hmr.antgroup-inc.cn/${namespace}/deepseek-v3:v1.0.2 --plain-http --output-remote --concurrency 16
```

#### 指定 source 信息
若在构建时，需要将模型源信息自定义添加进模型镜像中，可通过以下命令指定，<font style="background-color:#FBDE28;">默认行为是当检测到模型当前目录下为 git 仓库或 zeta 仓库时会自动解析 source 相关信息，无需用户指定</font>，其他情况可按需指定。

```yaml
$ modctl build -f /path/to/modelfile/Modelfile /path/to/model \
-t hmr.antgroup-inc.cn/${namespace}/deepseek-v3:v1.0.2 --plain-http --output-remote --concurrency 16 --source-url https://huggingface.co/deepseek-ai/DeepSeek-R1 --source-revision 44effdfa8e727bc64ee7f
```

#### 追加/覆盖
在某些场景下，可能需要追加/修改已经构建并上传的模型中的某个文件，如 config.json, 如果再重新完整构建上传一次，成本会比较高，所以可以通过以下命令来追加/修改已构建的模型镜像。

```yaml
# 追加一个之前不存在的文件
$ modctl attach foo.txt -s registry.com/models/llama3:v1.0.0 -t registry.com/models/llama3:v1.0.1 --output-remote

# 覆盖/修改原来已存在的文件
$ modctl attach foo.txt -s registry.com/models/llama3:v1.0.0 -t registry.com/models/llama3:v1.0.1 --output-remote --force
```

## 下载
### Dragonfly 加速下载
**支持的环境**

[https://yuque.antfin.com/baimo.qwb/mecxix/otbfdnogfowuy6gt?singleDoc#](https://yuque.antfin.com/baimo.qwb/mecxix/otbfdnogfowuy6gt?singleDoc#)

如果需要新增集群支持，咨询[@百蓦](https://yuque.antfin.com/baimo.qwb)[@肃晗](https://yuque.antfin.com/suhan.zcy)。

#### Pod 增加 NODE_IP ENV
POD 启动时，**<font style="color:#FFFFFF;background-color:#C99103;">增加 NODE_IP ENV 到 Pod</font>**:

```yaml
spec:
  containers:
  - env:
    - name: NODE_IP
      valueFrom:
        fieldRef:
          fieldPath: status.hostIP
```

#### Pod 内测试 NODE_IP 是否可用
```shell
$ curl -v $NODE_IP:4003/healthy
*   Trying 30.230.75.220:4003...
* Connected to 30.230.75.220 (30.230.75.220) port 4003
> GET /healthy HTTP/1.1
> Host: 30.230.75.220:4003
> User-Agent: curl/8.4.0
> Accept: */*
>
< HTTP/1.1 200 OK
< content-length: 0
< date: Tue, 17 Dec 2024 09:38:36 GMT
<
* Connection #0 to host 30.230.75.220 left intact
```

#### 登录
没有登陆的需要先[登录](#PlGag)。

#### 下载模型
`/tmp/deepseek-v3/`即为需要下载模型的指定目录, `${namespace}`为申请的 Namespace。

```shell
$ modctl pull hmr.sa128.alipay.com/${namespace}/deepseek-v3:v1.0.2 --proxy http://$NODE_IP:4001 --plain-http --insecure --extract-from-remote --extract-dir /tmp/deepseek-v3/
Copying blob     sha256:1c05ac5a620306ecaec0e09e8adbbd4c25dc35b60825bef6a03c2aa4aab7939b skipped: already exists
Copying blob     sha256:43105779bb4ceb010f28b3b6dc9a360455530209229948abc25900f18e21006c skipped: already exists
Copying blob     sha256:011ff244caff15289524da3b802f90637f5720e6887e2d5de35d89edcc16c270 skipped: already exists
Copying blob     sha256:0231d4dd488b86c56c657773598c8dadf0b9fc9563e8d630ba0909389d7f57d4 skipped: already exists
Copying blob     sha256:9efe10adeb9c76db166da2e476035721d025bd51cace8c4ddf15c25c094b7f74 | 988 MB | done
Copying blob     sha256:d5ca6ac7ac96ed52322cca8b860cd783114842d205b4084d14ef924214b6df6f | 7.0 MB | done
Copying blob     sha256:373d09d0b4d7299829440fdedd5d783268baf031454d35e08ee1f8aea5bbaf00 | 9.2 kB | done
Copying blob     sha256:7ff052b041a903c08f4335b9b3d9bd4260f342aa3d7cd570a682a6a588bf4e33 | 2.8 MB | done
Copying config   sha256:73ec349249e76abe2bde801d664aecf42cbb4adee66da0b1a01792aa78632a33 | 44 B | done
Copying manifest sha256:94c1c9dc014f65dd96ca76557d82d863d538855d60e2f52ad3f9e9bc66feccda | 3.0 kB | done
Successfully pulled model artifact: hmr.sa128.alipay.com/${namespace}/deepseek-v3:v1.0.2
```



### 普通下载
#### 登录
没有登陆的需要先[登录](#PlGag)。

#### 下载模型
`/tmp/deepseek-v3/`即为需要下载模型的指定目录, `${namespace}`为申请的 Namespace。

```yaml
$ modctl pull hmr.sa128.alipay.com/${namespace}/deepseek-v3:v1.0.2 --plain-http --insecure --extract-from-remote --extract-dir /tmp/deepseek-v3/
Copying blob     sha256:1c05ac5a620306ecaec0e09e8adbbd4c25dc35b60825bef6a03c2aa4aab7939b skipped: already exists
Copying blob     sha256:43105779bb4ceb010f28b3b6dc9a360455530209229948abc25900f18e21006c skipped: already exists
Copying blob     sha256:011ff244caff15289524da3b802f90637f5720e6887e2d5de35d89edcc16c270 skipped: already exists
Copying blob     sha256:0231d4dd488b86c56c657773598c8dadf0b9fc9563e8d630ba0909389d7f57d4 skipped: already exists
Copying blob     sha256:9efe10adeb9c76db166da2e476035721d025bd51cace8c4ddf15c25c094b7f74 | 988 MB | done
Copying blob     sha256:d5ca6ac7ac96ed52322cca8b860cd783114842d205b4084d14ef924214b6df6f | 7.0 MB | done
Copying blob     sha256:373d09d0b4d7299829440fdedd5d783268baf031454d35e08ee1f8aea5bbaf00 | 9.2 kB | done
Copying blob     sha256:7ff052b041a903c08f4335b9b3d9bd4260f342aa3d7cd570a682a6a588bf4e33 | 2.8 MB | done
Copying config   sha256:73ec349249e76abe2bde801d664aecf42cbb4adee66da0b1a01792aa78632a33 | 44 B | done
Copying manifest sha256:94c1c9dc014f65dd96ca76557d82d863d538855d60e2f52ad3f9e9bc66feccda | 3.0 kB | done
Successfully pulled model artifact: hmr.sa128.alipay.com/${namespace}/deepseek-v3:v1.0.2
```

### 部分下载
在某些场景下，可能并不需要使用到模型中的所有文件，只需要使用到部分文件，那么也支持通过如下命令来下载部分文件。

```yaml
$ modctl fetch registry.com/models/llama3:v1.0.0 --output /path/to/extract --patterns '*.json'
```

> ⚠️ 如果想匹配子目录中的文件，则需要把目录层级写出来，例如想匹配一级目录下的 json 文件需要写`*/*.json`
>
