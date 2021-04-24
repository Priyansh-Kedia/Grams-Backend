import paramiko

def main():
    host = "103.26.216.132"
    port = 22
    username = "inweon"
    password = "cm2017"

    command = "ls"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)

    stdin, stdout, stderr = ssh.exec_command(command)
    lines = stdout.readlines()
    print(lines)

if __name__ == "__main__":
    main()
