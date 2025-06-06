"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from pycrazyswarm import Crazyswarm


TAKEOFF_DURATION = 2.5
HOVER_DURATION = 2.0


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf1 = swarm.allcfs.crazyflies[0]
    # cf2 = swarm.allcfs.crazyflies[1]

    cf1.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    # cf2.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    cf1.land(targetHeight=0.04, duration=2.5)
    # cf2.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)


if __name__ == "__main__":
    main()
