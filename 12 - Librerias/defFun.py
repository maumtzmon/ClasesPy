from subprocess import call as call
def print_many():
    print("hello")
    print("Hola")
    print("Ciao")
    print("Óla")

def addition():
    sum = 1+1
    print("1 +1 = %s" % sum)

def tmp_space():
    tmp_usage = "ls"
    tmp_arg = "-l"
    path = "/tmp"
    print("Space used in /tmp directory")
    call([tmp_usage, tmp_arg, path])

def main():
    print_many()
    addition()
    tmp_space()


if __name__ == "__main__":  #very importan to use this file as a library
    main()