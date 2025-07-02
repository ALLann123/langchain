#!/usr/bin/python3
import paramiko 

def connected_kali(command):
#create an object/instance
    ssh_client=paramiko.SSHClient()

    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 

    ssh_client.connect(hostname='192.168.1.101', port=22, username='', password='')

    try:
         stdin, stdout, stderr=ssh_client.exec_command(command)

         output=stdout.read().decode()

         error=stderr.read().decode()

         if output:
              return output
         if error:
              return error
    finally:
         ssh_client.close()


