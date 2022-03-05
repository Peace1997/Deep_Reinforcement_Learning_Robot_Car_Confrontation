import numpy as np
import math
import Box2D
from Box2D.b2 import (fixtureDef, polygonShape, )
from utils import UserData, COLOR_BLACK

SIZE = 0.001
BULLET_BOX = [(0,0),(2,-0.08),(2,0.08)]
RADIUS_START = 0.9


class Projectile:
    def __init__(self, world):
        self.__world = world
        self.__projectile = {}
        self.__ctr = 1
        self.__fixture_bullet = [fixtureDef(
            shape=polygonShape(vertices=BULLET_BOX),
            categoryBits=0x02,
            maskBits=0xFD,
            density=1e-6
        )]

    def shoot(self, robot, init_angle, init_pos):
        angle = init_angle
        x, y = init_pos
        x += math.cos(angle) * RADIUS_START
        y += math.sin(angle) * RADIUS_START
        userData = UserData("bullet", self.__ctr)
        self.__fixture_bullet[0].userData = userData
        projectile = self.__world.CreateDynamicBody(
            position=(x, y),
            angle=angle,
            fixtures=self.__fixture_bullet,
        )
        #bullet.bullet = True
        if robot.robot_id == 0:
            # projectile.color = (0.5,0.8,0.4) #green
            projectile.color = (0.9, 0.5, 0.4)
        if robot.robot_id == 1:
            #projectile.color = (0.5,0.8,0.4)
            projectile.color = (0.5, 0.7, 0.9)
        #bullet.userData = userData
        projectile.linearVelocity = (math.cos(angle)*5, math.sin(angle)*5)
        self.__projectile[self.__ctr] = projectile
        self.__ctr += 1

    def draw(self, viewer):
        for obj in self.__projectile.values():
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=obj.color)

    def destroyById(self, bullet_id):
        body = self.__projectile.pop(bullet_id, None)
        if body is not None:
            self.__world.DestroyBody(body)

    def destroy(self):
        for bullet in self.__projectile.values():
            self.__world.DestroyBody(bullet)
        self.__projectile = {}
