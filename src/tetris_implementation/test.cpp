
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>

int main()
{
    const char* name = "pipe_named";
    const mode_t permission = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH; // 644

   if (mkfifo(name, permission) != 0)
   {

       exit(EXIT_FAILURE);
   }

   const int fd = open(name, O_RDONLY);

    char buffer[100];
    ssize_t bytesRead;

    while ((bytesRead = read(fd, &buffer, sizeof(buffer))) > 0) {
        write(STDOUT_FILENO, &buffer, bytesRead);
    }

   close(fd);
   unlink(name);
}