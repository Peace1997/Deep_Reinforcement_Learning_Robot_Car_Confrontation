# ICRA 2019 Battlefield simulator.
# Based on Top-down car dynamics simulation from OpenAI Gym.
import math
import random
import sys

import Box2D
import gym
import numpy as np
import pyglet
from gym import spaces
from gym.utils import EzPickle, colorize, seeding
from pyglet import gl


from battlefield.body.obstacle import ICRALayout
from battlefield.body.robot import Robot
from battlefield.body.projectile import Projectile
from battlefield.referee.contact import ContactListener
#from battlefield.referee.buff import AreaBuff
#from battlefield.referee.supply import AreaSupply
from battlefield.sensor.capture import callback_capture
from utils import *
from agent.Agent import redAgent,blueAgent
WINDOW_W = 1200
WINDOW_H = 1000

SCALE = 40.0        # Track scale
PLAYFIELD = 400/SCALE  # Game over boundary
FPS = 30
ZOOM = 2.7        # Camera zoom

SCAN_RANGE = 10  # m


class ICRABattleField(gym.Env, EzPickle):

    __pos_safe = [
        [0.5, 0.5], [0.5, 2.0], [0.5, 3.0], [0.5, 4.5],  # 0 1 2 3
        [1.5, 0.5], [1.5, 3.0], [1.5, 4.5],             # 4 5 6
        [2.75, 0.5], [2.75, 2.0], [2.75, 3.0], [2.75, 4.5],  # 7 8 9 10
        [4.0, 1.75], [4.0, 3.25],                         # 11 12
        [5.25, 0.5], [5.25, 2.0], [5.25, 3.0], [5.25, 4.5],  # 13 14 15 16
        [6.5, 0.5], [6.5, 2.0], [6.5, 4.5],             # 17 18 19
        [7.5, 0.5], [7.5, 2.0], [7.5, 3.0], [7.5, 4.5]  # 20 21 22 23
    ]
    __id_pos_linked = [
        [1, 2, 3, 4], [0, 2, 3], [0, 1, 3, 5], [0, 1, 2, 6],
        [0, 7], [2, 9], [3, 10],
        [8, 9, 10, 4], [7, 9, 10, 11], [7, 8, 10, 5, 12], [7, 8, 9],
        [8, 14], [9, 15],
        [14, 15, 16, 17], [13, 15, 16, 18, 11, 11, 11, 11, 11], [
            13, 14, 16, 12, 12, 12, 12, 12], [13, 14, 15, 19],
        [13, 20], [14, 21], [16, 23],
        [21, 22, 23, 17], [20, 22, 23, 18], [20, 21, 23], [20, 21, 22, 19]
    ]

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.__contactListener_keepref = ContactListener(self)
        self.__world = Box2D.b2World(
            (0, 0), contactListener=self.__contactListener_keepref)
        self.viewer = None
        self.red_agent_num = 2 
        self.blue_agent_num = 1
        self.__robots = []
        self.__robot_name = [ID_R1,ID_R2,ID_B1]
        self.__obstacle = None
        #self.__area_buff = None
        self.__projectile = None
        #self.__area_supply = None
        self.__callback_autoaim = callback_capture()
        self.state_size = 120
        self.reward = []
      
        self.prev_reward = []
    

        self.red_agent = []
        self.blue_agent = []
        for i in range(self.red_agent_num):
            self.red_agent.append(redAgent())
        for i in range(self.blue_agent_num):
            self.blue_agent.append(blueAgent())
        
        self.state = []
        self.contact_wall_number = 0
        self.contact_robot_number = 0

        self.shared_reward  = True
        self.observation_space = []
        self.action_space = []

        for i in range(self.red_agent_num):
            self.observation_space.append(spaces.Box(low=np.zeros(self.state_size)-np.ones(self.state_size), high=np.ones(self.state_size), dtype=np.float32))
            self.action_space.append(spaces.Box(low=np.zeros(3)-np.ones(3),high=np.ones(3),dtype=np.float32))


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        for r in self.__robots:
            if r:
                r.destroy()
            r = None
        if self.__obstacle:
            self.__obstacle.destroy()
        self.__obstacle = None
        if self.__projectile:
            self.__projectile.destroy()
        self.__projectile = None

    def reset(self):
        self._destroy()
        self.reward = [0.0]*self.red_agent_num
        self.prev_reward = [0.0]*self.red_agent_num
        self.t = 0.0

        
        random_index = random.sample(range(len(self.__pos_safe)-1),self.red_agent_num + self.blue_agent_num)

        init_pos_0 = self.__pos_safe[random_index[0]]
        init_pos_1 = self.__pos_safe[random_index[1]]
        init_pos_2 = self.__pos_safe[random_index[2]]

        self.__R1 = Robot(self.__world, 0, init_pos_0, ID_R1)
        self.__R2 = Robot(self.__world, 0, init_pos_1, ID_R2)
        self.__B1 = Robot(self.__world, 0, init_pos_2, ID_B1)

        self.__robots = [self.__R1, self.__R2, self.__B1]
        self.__obstacle = ICRALayout(self.__world)
        self.__projectile = Projectile(self.__world)
        '''
        self.__area_buff = AreaBuff()
        self.__area_supply = AreaSupply()
        '''
        self.red_agent[0].pos = init_pos_0
        self.red_agent[1].pos = init_pos_1
        self.blue_agent[0].pos = init_pos_2
        #self.state.append([self.red_agent, self.blue_agent])
        self.state = []
        self.state = self.__concat_state()
        #self.actions = [Action(), Action()]

        #self.reward.append(0.0)

        return self.state
        #self.state[self.state_size:2*self.state_size]
        # return self.step(None)[0]

    def __step_contact(self):
        contact_bullet_sensor = self.__contactListener_keepref.collision_bullet_sensor
        # contact_bullet_robot = self.__contactListener_keepref.collision_bullet_robot
        contact_bullet_wall = self.__contactListener_keepref.collision_bullet_wall

        '''
        for bullet, robot in contact_bullet_robot:
            self.__projectile.destroyById(bullet.id)
            # if(self.__robots[robot.id].buff_left_time) > 0:
            #     self.__robots[robot.id].lose_health(25)
            # else:
            self.__robots[robot.id].lose_health(0)
        '''
        for bullet, robot in contact_bullet_sensor:
            self.__projectile.destroyById(bullet.id)
            self.__robots[robot.id].lose_health(1)
            
        for bullet in contact_bullet_wall:
            self.__projectile.destroyById(bullet.id)
        '''
        robot_wall = False
        robot_robot = False
        for robot in contact_robot_wall:
            if robot.id == 0:
                robot_wall = True
            # self.__robots[robot.id].lose_health(100)
        for robot in contact_robot_robot:
            if robot.id == 0
                robot_robot = True
            #self.__robots[robot.id].lose_health(10)
        
        self.__contactListener_keepref.clean()
        '''
    def _red_step_action(self, robot: Robot, red_agent: redAgent):
        # gas, rotate, transverse, rotate cloud terrance, shoot
        robot.move_ahead_back(red_agent.v_t)
        robot.move_left_right(red_agent.v_n)
        robot.turn_left_right(red_agent.angular)

        '''
        if int(self.t * FPS) % (60 * FPS) == 0:
            robot.refresh_supply_oppotunity()
        if action.supply > 0.99:
            action.supply = 0.0
            if robot.if_supply_available():
                robot.use_supply_oppotunity()
                if self.__area_supply.if_in_area(robot):
                    robot.supply()
        '''
        if red_agent.shoot > 0.99 and int(self.t*FPS) % (FPS/5) == 1:
            if(robot.if_left_projectile()):
                angle, pos = robot.get_gun_angle_pos()
                robot.shoot()
                #red_agent.shoot = 0.0
                self.__projectile.shoot(robot,angle, pos)

    def _blue_step_action(self, robot: Robot, blue_agent: blueAgent):
        # gas, rotate, transverse, rotate cloud terrance, shoot
        robot.move_ahead_back(blue_agent.v_t)
        robot.move_left_right(blue_agent.v_n)
        robot.turn_left_right(blue_agent.angular)

        '''
        if int(self.t * FPS) % (60 * FPS) == 0:
            robot.refresh_supply_oppotunity()
        if action.supply > 0.99:
            action.supply = 0.0
            if robot.if_supply_available():
                robot.use_supply_oppotunity()
                if self.__area_supply.if_in_area(robot):
                    robot.supply()
        '''
        if blue_agent.shoot > 0.99 and int(self.t*FPS) % (FPS/5) == 1:
            if(robot.if_left_projectile()):
                angle, pos = robot.get_gun_angle_pos()
                robot.shoot()
                #blue_agent.shoot = 0.0
                self.__projectile.shoot(robot,angle, pos)

    def _red_autoaim(self, robot: Robot,red_agent : redAgent):
        #detected = {}
        scan_distance, scan_type = [], []
        red_agent.detect = False
        for i in range(-30, 30, 1):
            angle, pos = robot.get_angle_pos()
            angle += i/180*math.pi
            p1 = (pos[0] + 0.2*math.cos(angle), pos[1] + 0.2*math.sin(angle))
            p2 = (pos[0] + SCAN_RANGE*math.cos(angle),
                  pos[1] + SCAN_RANGE*math.sin(angle))
            self.__world.RayCast(self.__callback_autoaim, p1, p2)
            scan_distance.append(self.__callback_autoaim.fraction)
            u = self.__callback_autoaim.userData
            if u is not None and u.type == "robot":
                scan_type.append(1)
                if -15 <= i <= 15:
                    if not red_agent.detect:
                        #robot.set_gimbal(angle)
                        distance = self.distance_detection()
                        if(distance <4):
                            red_agent.detect = True
            else:
                scan_type.append(0)
        red_agent.scan = scan_distance+scan_type

    def _blue_autoaim(self, robot: Robot,blue_agent: blueAgent):
        #detected = {}
        scan_distance, scan_type = [], []
        blue_agent.detect = False
        for i in range(-30, 30, 1):
            angle, pos = robot.get_angle_pos()
            angle += i/180*math.pi
            p1 = (pos[0] + 0.2*math.cos(angle), pos[1] + 0.2*math.sin(angle))
            p2 = (pos[0] + SCAN_RANGE*math.cos(angle),
                  pos[1] + SCAN_RANGE*math.sin(angle))
            self.__world.RayCast(self.__callback_autoaim, p1, p2)
            scan_distance.append(self.__callback_autoaim.fraction)
            u = self.__callback_autoaim.userData
            if u is not None and u.type == "robot":
                scan_type.append(1)
                if -15 <= i <= 15:
                    if not blue_agent.detect:
                        #robot.set_gimbal(angle)
                        distance = self.distance_detection()
                        if(distance <4):
                            blue_agent.detect = True
            else:
                scan_type.append(0)
        blue_agent.scan = scan_distance+scan_type

    def _update_red_robot_state(self, robot: Robot, red_agent: redAgent):
        red_agent.pos = robot.get_pos()
        red_agent.health = robot.get_health()
        red_agent.angle = robot.get_angle()
        #state.velocity = robot.get_velocity()
        #state.angular = robot.get_angular()

    def _update_blue_robot_state(self, robot: Robot, blue_agent: blueAgent):
        blue_agent.pos = robot.get_pos()
        blue_agent.health = robot.get_health()
        blue_agent.angle = robot.get_angle()

    def distance_detection(self):
        robot_pos = []
        for robot in self.__robots :
            robot_pos.append(robot.get_pos())
        difference_x = robot_pos[0][0] - robot_pos[1][0]
        difference_y = robot_pos[0][1] - robot_pos[1][1]
        distance = math.sqrt( (difference_x**2) + (difference_y**2) )
        return distance

    def step(self, red_action):
        ###### observe ######

        self._red_autoaim(self.__robots[ID_R1],self.red_agent[0])
        self._red_autoaim(self.__robots[ID_R2],self.red_agent[1])
        self._update_red_robot_state(self.__robots[0],self.red_agent[0])
        self._update_red_robot_state(self.__robots[1],self.red_agent[1])
        self._blue_autoaim(self.__robots[ID_B1],self.blue_agent[0])
        self._update_blue_robot_state(self.__robots[2], self.blue_agent[0])
        
        self.state = self.__concat_state()

        ID_B1_prev_health = self.__robots[ID_B1].get_health()
        ID_R1_prev_health = self.__robots[ID_R1].get_health()
        ID_R2_prev_health = self.__robots[ID_R2].get_health()

        ###### action ######
        self.red_agent[0].decode_action(red_action[0])
        self.red_agent[1].decode_action(red_action[1])
        blue_agent_action = [torch.from_numpy(self.action_space[0].sample())]
        self.blue_agent[0].decode_action(blue_agent_action[0])
        

        self._red_step_action(self.__robots[0], self.red_agent[0])
        self._red_step_action(self.__robots[1], self.red_agent[1])
        self.__robots[0].step(1.0/FPS)
        self.__robots[1].step(1.0/FPS)
        self._blue_step_action(self.__robots[2], self.blue_agent[0])
        self.__robots[2].step(1.0/FPS)

        self.__world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        ###### Referee ######
        self.__step_contact()
        '''
        for robot in self.__robots:
            self.__area_buff.detect(robot, self.t)
        '''
        ###### reward ######
        step_reward = 0

        done = False

        # First step without action, called from reset()
        contact_robot_wall = self.__contactListener_keepref.collision_robot_wall
        contact_robot_robot = self.__contactListener_keepref.collision_robot_robot
        ID_B1_now_health = self.__robots[ID_B1].get_health()
        ID_R1_now_health = self.__robots[ID_R1].get_health()
        ID_R2_now_health = self.__robots[ID_R2].get_health()
        if ID_B1_now_health < ID_B1_prev_health:
           step_reward += (0.2*(self.distance_detection()- 0.4))
           #step_reward += 0.1
        if (self.red_agent[0].detect) or (self.red_agent[1].detect):
            step_reward += 0.001
        
        if (ID_R1_now_health < ID_R1_prev_health) or (ID_R2_now_health < ID_R2_prev_health):
            step_reward -= 0.1
        # self.reward = (self.__robots[ID_R1].get_health() - self.__robots[ID_B1].get_health()) / 4000.0

        #self.reward += 10 * self.t * FPS
        #step_reward = self.reward - self.prev_reward
        # if self.red_agent.detect:
           # step_reward += 1/600

        if (self.__robots[ID_R1].get_health() <= 0) or (self.__robots[ID_R2].get_health() <= 0):
            done = True
            step_reward -= 3
        if self.__robots[ID_B1].get_health() <= 0:
            done = True
            step_reward += 5
        # if ep_len %1000 == 0:
        #     self.contact_wall_number = 0
        #     self.contact_robot_number = 0
        for robot in contact_robot_wall:
            if robot.id == 0:
                step_reward -= 0.01
            if robot.id == 1:
                step_reward -= 0.01
                    #step_reward -= 0.001 + (self.contact_wall_number - 100)*0.00001
        for robot in contact_robot_robot:
            if robot.id == 0:
                step_reward -= 0.01
            if robot.id == 1:
                step_reward -= 0.01
                    #step_reward -= 0.001 + (self.contact_robot_number - 100)*0.00001
        # if self.distance_detection()<1:
        #     step_reward -=0.01


        self.reward = [self.reward[0] + step_reward]*self.red_agent_num
        #self.reward = self.reward[0]
        #self.prev_reward = self.reward
        self.__contactListener_keepref.clean()
        return self.state,[step_reward]*self.red_agent_num, done, {}

    def __concat_state(self):
        red_state = []
        # ID_R1_position_x =self.red_agent.pos[0]/8.0
        # ID_R1_position_y =self.red_agent.pos[1]/6
        # ID_R1_angle = self.red_agent.angle / 2 * math.pi
        # ID_R1_detect =self.red_agent.detect
        # ID_R1_health = self.red_agent.health/20

        #ID_R1_scan = self.red_agent.scan.copy()

        # ID_R1_position_x =np.array(self.red_agent.pos[0]/8.0,dtype = np.float32)
        # ID_R1_position_y =np.array(self.red_agent.pos[1]/6,dtype=np.float32)
        # ID_R1_angle = np.array(self.red_agent.angle / 2 * math.pi, dtype=np.float32)
        # ID_R1_detect = np.array(self.red_agent.detect,dtype=np.float32)
        # ID_R1_health = np.array(self.red_agent.health/20,dtype=np.float32)
        
        ID_R1_scan = np.array(self.red_agent[0].scan.copy(),dtype=np.float32)
        ID_R2_scan = np.array(self.red_agent[1].scan.copy(),dtype=np.float32)
        # for i in range(len(ID_R1_scan)):
        #     state.append(ID_R1_scan[i])

        red_state.append(ID_R1_scan)
        red_state.append(ID_R2_scan)
        # ID_B1_position_x = self.blue_agent.pos[0]/8.0
        # ID_B1_position_y = self.blue_agent.pos[1]/6
        # ID_B1_angle = self.blue_agent.angle/2*math.pi
        # ID_B1_detect = self.blue_agent.detect
        # ID_B1_health = self.blue_agent.health/20
        # ID_B1_scan = self.blue_agent.scan.copy()

        # ID_B1_position_x = np.array(self.blue_agent.pos[0]/8.0,dtype=np.float32)
        # ID_B1_position_y = np.array(self.blue_agent.pos[1]/6,dtype = np.float32)
        # ID_B1_angle = np.array(self.blue_agent.angle/2*math.pi, dtype=np.float32)
        # ID_B1_detect = np.array(self.blue_agent.detect, dtype=np.float32)
        # ID_B1_health = np.array(self.blue_agent.health/20,dtype=np.float32)
        
        #####ID_B1_scan = np.array(self.blue_agent.scan.copy(),dtype=np.float32)

        # state.append(ID_R1_position_x)
        # state.append(ID_R1_position_y)
        # state.append(ID_B1_position_x)
        # state.append(ID_B1_position_y)
        # state.append(ID_R1_angle)
        # state.append(ID_R1_detect)
        # state.append(ID_R1_health)



        #state.append(ID_B1_angle)
        # state.append(ID_B1_detect)
        # state.append(ID_B1_health)
        
        # for i in range(len(ID_B1_scan)):
        #     state.append(ID_B1_scan[i])


        return red_state
    @staticmethod
    def get_gl_text(x, y):
        return pyglet.text.Label('0000', font_size=16, x=x, y=y,
                                 anchor_x='left', anchor_y='center',
                                 color=(255, 255, 255, 255))

    def render(self, mode='god'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.time_label = self.get_gl_text(20, WINDOW_H * 5.0 / 40.0)
            self.score_label = self.get_gl_text(520, WINDOW_H * 2.5 / 40.0)
            self.health_label = self.get_gl_text(520, WINDOW_H * 3.5 / 40.0)
            self.projectile_label = self.get_gl_text(
                520, WINDOW_H * 4.5 / 40.0)
            '''
            self.buff_left_time_label = self.get_gl_text(
                520, WINDOW_H * 5.5 / 40.0)
            self.buff_stay_time = self.get_gl_text(520, WINDOW_H * 6.5 / 40.0)
            '''
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        zoom = ZOOM*SCALE
        scroll_x = 4.0
        scroll_y = 0.0
        angle = 0
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) -
                          scroll_y*zoom*math.sin(angle)),
            WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)))

        self.__obstacle.draw(self.viewer)
        if mode == 'god':
            for robot in self.__robots:
                robot.draw(self.viewer)
        elif mode == "fps":
            self.__robots[ID_R1].draw(self.viewer)
            self.__robots[ID_B1].draw(self.viewer)
        self.__projectile.draw(self.viewer)

        arr = None
        win = self.viewer.window
        if mode != 'state_pixels':
            win.switch_to()
            win.dispatch_events()

        win.clear()
        t = self.transform
        gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
        t.enable()
        self._render_background()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        t.disable()
        self._render_indicators(WINDOW_W, WINDOW_H)
        win.flip()

        self.viewer.onetime_geoms = []
        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _render_background(self):
        # back_image = pyglet.resource.image('background.jpeg')
        # back_image.blit(0,0)
        # image = pyglet.image.load("background.jpeg")
        # background_sprite = pyglet.sprite.Sprite(image)
        # background_sprite.draw()
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.9, 0.9, 0.9, 0.9)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        #gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        # k = PLAYFIELD/20.0
        # for x in range(-20, 20, 2):
        #     for y in range(-20, 20, 2):
        #         gl.glVertex3f(k*x + k, k*y + 0, 0)
        #         gl.glVertex3f(k*x + 0, k*y + 0, 0)
        #         gl.glVertex3f(k*x + 0, k*y + k, 0)
        #         gl.glVertex3f(k*x + k, k*y + k, 0)
        gl.glEnd()
        #self.__area_buff.render(gl)
        #self.__area_supply.render(gl)
        #self.__area_buff.render(gl)
        #self.__area_supply.render(gl)

    def _render_indicators(self, W, H):
        self.time_label.text = "Time: {} s".format(int(self.t))
        self.score_label.text = "Score: %04i" % self.reward[0]
        self.health_label.text = "Health:  RedCar1 : {}  RedCar2 : {}  BlueCar: {} ".format(
            self.__robots[ID_R1].get_health(), self.__robots[ID_R2].get_health(),self.__robots[ID_B1].get_health())
        self.projectile_label.text = "Bullets: RedCar1 : {} RedCar2 : {} BlueCar: {}".format(
            self.__robots[ID_R1].get_left_projectile(),self.__robots[ID_R2].get_left_projectile(),self.__robots[ID_B1].get_left_projectile()
        )

        '''
        self.buff_stay_time.text = 'Buff Stay Time: Red {}s, Blue {}s'.format(
            int(self.__area_buff.get_single_buff(GROUP_RED).get_stay_time()),
            int(self.__area_buff.get_single_buff(GROUP_BLUE).get_stay_time()))
        self.buff_left_time_label.text = 'Buff Left Time: Red {}s, Blue {}s'.format(
            int(self.__robots[ID_R1].buff_left_time),
            int(self.__robots[ID_B1].buff_left_time))
        '''
        self.time_label.draw()
        self.score_label.draw()
        self.health_label.draw()
        self.projectile_label.draw()
        '''
        self.buff_stay_time.draw()
        self.buff_left_time_label.draw()
        '''

