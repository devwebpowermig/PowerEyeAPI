from PowerEye_v03 import joint_tracker


def main():

    new_tracker = joint_tracker(source="video_01.avi")
    new_tracker.imshow()


if __name__ == "__main__":

    main()

    while (True):
        res = input('\nDeseja executar o algoritmo novamente? [s/_]: ')

        if (res == 's'):
            main()
        else:
            break

    quit()
