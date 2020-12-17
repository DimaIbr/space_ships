import pygame
import random
import numpy as np
from time import sleep

#FPS 1000/turn_delay
turn_delay=10

boot_shoot_delay=25#количество ходов

#размеры окна
width_win=1000
heigth_win=1000

#Стартовая позиция игрока
start_point_x=100
start_point_y=100

#Базовые размеры объектов
person_size=20
person_defolt=20
boolet=person_defolt//2
comet_size=person_defolt//2

#Условия победы
winner_size=person_defolt*5
bots_remain_winner=1

#базовые сокрости объектов
speed_move=10
speed_shoot=speed_move*4
speed_comet=speed_move

#Базовые очки за действия
shoot_k=2
kill_komet_k=3

#Базовое количество ботов
number_of_bots=24

bots_remain=number_of_bots

win=pygame.display.set_mode((width_win,heigth_win))
pygame.display.set_caption("Space ships")#Исправить потом

glabal_names=[]

endgamebad_i=pygame.image.load('endgame.png')
endgamebad_rect=endgamebad_i.get_rect(bottomright=(width_win,heigth_win))

n=11#число входных данных
m=24 #число нейронов в скротом слое
k=5 #число выходов

population={}
results={}
dead_bots={}

for i in range(number_of_bots):
    cromosome = np.zeros(n * m + m * k)
    for j in range(len(cromosome)):
        cromosome[j] = random.random()
    population[i]=cromosome
    results[i]=0


def rand_color_rgb():
    #Задание рандомного цвета боту
    return (random.randint(1,255),random.randint(1,255),random.randint(1,255))

def sigmoid(x):
    #функция активации - сигмоида
    return 1/(1+np.exp(-x))

class game_object:
    def __init__(self,position_x,position_y,size,speed,color,type_ob,parent='defolt',name='defoult',face='UP'):
        self.name=name
        self.x=position_x
        self.y=position_y
        self.dead=0
        self.size=size
        self.speed=speed
        self.color=color
        self.type_ob=type_ob
        self.life_turn=0
        self.face=face
        self.turn_till_last_shoot=0
        self.parent=parent


    def is_dead_f(self):
        if self.x<0 and self.y<0:
            self.dead=1
            return 1
        else:
            return 0

class neuron:
    def __init__(self, weights, bias=0):
        self.weights = weights #веса
        #self.bias = bias #смещение

    def feedforward(self, inputs):
        # Умножаем входы на веса, прибавляем порог, затем используем функцию активации
        total = np.dot(self.weights, inputs)# + self.bias
        return sigmoid(total)

def shooted_obj(objects_in_game, x,y,size_shooted,size,speed,face,name_parent):
    if face=='UP':
        x+=(size_shooted-size)//2
    elif face=='DOWN':
        y+=size_shooted-size
        x += (size_shooted - size) // 2
    elif face=='LEFT':
        y+= (size_shooted - size) // 2
    else:
        y+= (size_shooted - size) // 2
        x+= size_shooted-size
    boolet_ad=game_object(x,y,size,speed,rand_color_rgb(),'boolet',face=face,parent=name_parent)
    objects_in_game['boolets'].append(boolet_ad)

def wall_selection():
    walls=['UP','DOWN','LEFT','RIGHT']
    return walls[random.randint(0,3)]

def spawn_comet(objects_in_game):
    wall=wall_selection()
    fullspeed=speed_comet*random.randint(1,3)
    if wall=='UP':
        y=0
        x=random.randint(1,width_win)
        speed_comet_x=random.randint(0,fullspeed//speed_comet)*speed_comet
        speed_comet_y=fullspeed-speed_comet_x
    elif wall=='DOWN':
        y=heigth_win
        x=random.randint(1,width_win)
        speed_comet_x = random.randint(0, fullspeed // speed_comet) * speed_comet
        speed_comet_y = -(fullspeed - speed_comet_x)
    elif wall=='LEFT':
        x=0
        y=random.randint(1,heigth_win)
        speed_comet_y = random.randint(0, fullspeed // speed_comet) * speed_comet
        speed_comet_x = fullspeed - speed_comet_y
    else:
        x=width_win
        y=random.randint(1,heigth_win)
        speed_comet_y = random.randint(0, fullspeed // speed_comet) * speed_comet
        speed_comet_x = -(fullspeed - speed_comet_y)
    objects_in_game['comets'].append(game_object(x,y,comet_size,[speed_comet_x,speed_comet_y],(192,192,192),'comet'))

def chek_space(x,y,size,objects_in_game):
    for j in objects_in_game['comets']:
        if ((x + size >= j.x and x <= j.x) and (y + size >= j.y and y <= j.y)or (x+size>=j.x and  x<=j.x and y+size>=j.y+j.size and y<=j.y+j.size)) and x > 0 and y > 0:
            return 0
    for j in objects_in_game['bots']:
        if (((x + size >= j.x and x <= j.x) and (y + size >= j.y and y <= j.y))or (x+size>=j.x and  x<=j.x and y+size>=j.y+j.size and y<=j.y+j.size)) and x > 0 and y > 0:
            return 0
    for j in objects_in_game['player']:
        if ((x + size >= j.x and x <= j.x) and (y + size >= j.y and y <= j.y)or (x+size>=j.x and  x<=j.x and y+size>=j.y+j.size and y<=j.y+j.size)) and x > 0 and y > 0:
            return 0
    for j in objects_in_game['boolets']:
        if ((x + size >= j.x and x <= j.x) and (y + size >= j.y and y <= j.y)or (x+size>=j.x and  x<=j.x and y+size>=j.y+j.size and y<=j.y+j.size)) and x > 0 and y > 0:
            return 0
    return 1

def spawn_bots(objects_in_game, number):
    for i in range(number):
        while True:
            x=random.randint(5,width_win-person_size-5)
            y=random.randint(5,heigth_win-person_size-5)
            if chek_space(x,y,person_size,objects_in_game):
                break
        objects_in_game['bots'].append(game_object(x,y,person_size,speed_move,rand_color_rgb(), 'bot', name=i))

def action_object(x_coor,y_coor,speed_object,size_object,face_object, act_object,name_object,dead,objects_in_game):
    #Движение объекта (любого)
    if act_object=='LEFT':
        if x_coor>speed_object:
            x_coor-=speed_object
            face_object='LEFT'
        else:
            dead=1
    if act_object=='RIGHT':
        if x_coor<width_win-size_object-speed_object:
            x_coor+=speed_object
            face_object='RIGHT'
        else:
            dead=1
    if act_object=='UP':
        if y_coor>speed_object:
            y_coor-=speed_object
            face_object='UP'
        else:
            dead=1
    if act_object=='DOWN':
        if y_coor<heigth_win-size_object-speed_object:
            y_coor+=speed_object
            face_object='DOWN'
        else:
            dead=1
    if act_object == 'SHOOT':
        shooted_obj(objects_in_game,x_coor,y_coor,size_object,boolet,speed_shoot,face_object,name_object)
    return x_coor, y_coor, face_object,dead

def comet_movement(objects_in_game,x,y,speed,size,dead):
    if speed[0]>0:
        x_move='RIGHT'
    else:
        x_move='LEFT'
    if speed[1]>0:
        y_move='DOWN'
    else:
        y_move='UP'
    x,y,face,dead = action_object(x,y,abs(speed[0]),size,'UP',x_move,'default',dead,objects_in_game)
    x, y, face,dead = action_object(x, y, abs(speed[1]), size, 'UP', y_move,'default',dead, objects_in_game)
    return x,y,dead

def key_interpritation(key_pressed):
    #Интерпритация нажатий кнопок игрока, чтобы передать её в фунцкию move_object
    if key_pressed[pygame.K_LEFT] or key_pressed[pygame.K_a]:
        return 'LEFT'
    if key_pressed[pygame.K_RIGHT] or key_pressed[pygame.K_d]:
        return 'RIGHT'
    if key_pressed[pygame.K_UP] or key_pressed[pygame.K_w]:
        return 'UP'
    if key_pressed[pygame.K_DOWN] or key_pressed[pygame.K_s]:
        return 'DOWN'
    if key_pressed[pygame.K_SPACE]:
        return 'SHOOT'

def del_dead_objects(objects_in_game,dead_bots):
    for i in objects_in_game['comets']:
        if i.is_dead_f() or i.dead:
            objects_in_game['comets'].pop(objects_in_game['comets'].index(i))
    for i in objects_in_game['bots']:
        if i.is_dead_f() or i.dead:
            dead_bots[i.name]=objects_in_game['bots'].pop(objects_in_game['bots'].index(i))
        else:
            i.life_turn+=1
    for i in objects_in_game['player']:
        if i.is_dead_f() or i.dead:
            objects_in_game['player'].pop(objects_in_game['player'].index(i))
    for i in objects_in_game['boolets']:
        if i.is_dead_f() or i.dead:
            objects_in_game['boolets'].pop(objects_in_game['boolets'].index(i))

def check_collision(i, objects_in_game):
    for j in objects_in_game['comets']:
        if i != j and ((i.x + i.size >= j.x and i.x <= j.x) and (i.y + i.size >= j.y and i.y <= j.y)or (i.x+i.size>=j.x and  i.x<=j.x and i.y+i.size>=j.y+j.size and i.y<=j.y+j.size)) and i.x > 0 and i.y > 0:
            i.dead = 1
            j.dead = 1
    for j in objects_in_game['bots']:
        if i != j and ((i.x + i.size >= j.x and i.x <= j.x) and (i.y + i.size >= j.y and i.y <= j.y)or (i.x+i.size>=j.x and  i.x<=j.x and i.y+i.size>=j.y+j.size and i.y<=j.y+j.size)) and i.x > 0 and i.y > 0:
            i.dead = 1
            j.dead = 1
    for j in objects_in_game['player']:
        if i != j and ((i.x + i.size >= j.x and i.x <= j.x) and (i.y + i.size >= j.y and i.y <= j.y)or (i.x+i.size>=j.x and  i.x<=j.x and i.y+i.size>=j.y+j.size and i.y<=j.y+j.size)) and i.x > 0 and i.y > 0:
            i.dead = 1
            j.dead = 1
    for j in objects_in_game['boolets']:
          if i != j and ((i.x + i.size >= j.x and i.x <= j.x) and (i.y + i.size >= j.y and i.y <= j.y)or (i.x+i.size>=j.x and  i.x<=j.x and i.y+i.size>=j.y+j.size and i.y<=j.y+j.size)) and i.x > 0 and i.y > 0 and j.parent!=i.name:
            if i.type_ob=='comet':
                if j.parent=='player':
                    objects_in_game['player'][0].size+=comet_size
                else:
                    for k in objects_in_game['bots']:
                        if k.name == j.parent:
                            k.size += comet_size
            elif i.type_ob=='bot':
                if j.parent=='player':
                    objects_in_game['player'][0].size+=i.size//2
                else:
                    for k in objects_in_game['bots']:
                        if k.name==j.parent:
                            k.size+=i.size//2
            i.dead = 1
            j.dead = 1

def del_collision_objects(objects_in_game):
    for i in objects_in_game['comets']:
        check_collision(i,objects_in_game)
    for i in objects_in_game['bots']:
        check_collision(i,objects_in_game)
    for i in objects_in_game['player']:
        check_collision(i,objects_in_game)
    for i in objects_in_game['boolets']:
        check_collision(i,objects_in_game)

def sensors(objects_in_game,object_g):

    default=10000

    def sensorleft(object_g):#y=object_g.y object_g.x>x
        min_o=default
        for i in objects_in_game['bots']:
            if (object_g.y<=i.y+i.size and object_g.y>=i.y or object_g.y+object_g.size<=i.y+i.size and object_g.y+object_g.size>=i.y ) and object_g.x>i.x and object_g!=i:
                if object_g.x-i.x<min_o:
                    min_o=object_g.x-i.x
        for i in objects_in_game['player']:
            if (object_g.y<=i.y+i.size and object_g.y>=i.y or object_g.y+object_g.size<=i.y+i.size and object_g.y+object_g.size>=i.y ) and object_g.x>i.x :
                if object_g.x - i.x < min_o:
                    min_o = object_g.x - i.x
        for i in objects_in_game['comets']:
            if (object_g.y<=i.y+i.size and object_g.y>=i.y or object_g.y+object_g.size<=i.y+i.size and object_g.y+object_g.size>=i.y ) and object_g.x>i.x :
                if object_g.x - i.x < min_o:
                    min_o = object_g.x - i.x
        if min_o==default:
            min_o=0
        return min_o
    def sensorright(object_g):#y=object_g.y object.x<x
        min_o = default
        for i in objects_in_game['bots']:
            if (object_g.y <= i.y + i.size and object_g.y >= i.y or object_g.y+object_g.size <= i.y + i.size and object_g.y+object_g.size >= i.y) and object_g.x<i.x and object_g != i:
                if -object_g.x+i.x  < min_o:
                    min_o = -object_g.x+i.x
        for i in objects_in_game['player']:
            if (object_g.y <= i.y + i.size and object_g.y >= i.y or object_g.y+object_g.size <= i.y + i.size and object_g.y+object_g.size >= i.y) and object_g.x<i.x :
                if -object_g.x+i.x < min_o:
                    min_o = -object_g.x+i.x
        for i in objects_in_game['comets']:
            if (object_g.y <= i.y + i.size and object_g.y >= i.y or object_g.y+object_g.size <= i.y + i.size and object_g.y+object_g.size >= i.y) and object_g.x<i.x :
                if -object_g.x+i.x  < min_o:
                    min_o = -object_g.x+i.x
        if min_o==default:
            min_o=0
        return min_o
    def sensordown(object_g):#x=object_g.x object.y<y
        min_o = default
        for i in objects_in_game['bots']:
            if (object_g.x <= i.x + i.size and object_g.x >= i.x or object_g.x+object_g.size <= i.x + i.size and object_g.x+object_g.size >= i.x) and object_g.y < i.y and object_g != i:
                if -object_g.y + i.y < min_o:
                    min_o = -object_g.y + i.y
        for i in objects_in_game['player']:
            if (object_g.x <= i.x + i.size and object_g.x >= i.x or object_g.x+object_g.size <= i.x + i.size and object_g.x+object_g.size >= i.x) and object_g.y < i.y :
                if -object_g.y + i.y < min_o:
                    min_o = -object_g.y + i.y
        for i in objects_in_game['comets']:
            if (object_g.x <= i.x + i.size and object_g.x >= i.x or object_g.x+object_g.size <= i.x + i.size and object_g.x+object_g.size >= i.x) and object_g.y < i.y :
                if -object_g.y + i.y < min_o:
                    min_o = -object_g.y + i.y
        if min_o==default:
            min_o=0
        return min_o
    def sensorup(object_g):#x=object_g.x object_g.y>y
        min_o = default
        for i in objects_in_game['bots']:
            if (object_g.x <= i.x + i.size and object_g.x >= i.x or object_g.x+object_g.size <= i.x + i.size and object_g.x+object_g.size >= i.x) and object_g.y > i.y and object_g != i:
                if object_g.y - i.y < min_o:
                    min_o = object_g.y - i.y
        for i in objects_in_game['player']:
            if (object_g.x <= i.x + i.size and object_g.x >= i.x or object_g.x+object_g.size <= i.x + i.size and object_g.x+object_g.size >= i.x) and object_g.y > i.y:
                if object_g.y - i.y < min_o:
                    min_o = object_g.y - i.y
        for i in objects_in_game['comets']:
            if (object_g.x <= i.x + i.size and object_g.x >= i.x or object_g.x+object_g.size <= i.x + i.size and object_g.x+object_g.size >= i.x) and object_g.y > i.y:
                if object_g.y - i.y < min_o:
                    min_o = object_g.y - i.y
        if min_o==default:
            min_o=0
        return min_o
    sensors_ob={'RIGHT':0,'LEFT':0,'UP':0,'DOWN':0}
    sensors_ob['RIGHT']=sensorright(object_g)
    sensors_ob['LEFT']=sensorleft(object_g)
    sensors_ob['UP']=sensorup(object_g)
    sensors_ob['DOWN']=sensordown(object_g)

    for key_s in sensors_ob:
        if key_s == 'UP':
            endy = 0
            endx = object_g.x
        elif key_s == 'DOWN':
            endy = heigth_win
            endx = object_g.x
        elif key_s == 'LEFT':
            endy = object_g.y
            endx = 0
        else:
            endy = object_g.y
            endx = width_win
        # pygame.draw.line(win, (255, 255, 255), (object_g.x, object_g.y), (endx, endy), 2)
        # if sensors_ob[key_s]!=0:
        #     pygame.draw.line(win,(255,0,0),(object_g.x,object_g.y),(endx,endy),2)
    return sensors_ob

def choose_comand(num):
    if num==0:
        return 'UP'
    elif num==1:
        return  'LEFT'
    elif num==2:
        return 'DOWN'
    elif num==3:
        return 'RIGHT'
    else:
        return 'SHOOT'

def face_nap(num):
    if num=='UP':
        return 0
    elif num=='LEFT':
        return  1
    elif num=='DOWN':
        return 2
    else:
        return 3

def cromosome_to_weights(cromosome,n,m,k):
    weight_hiden=np.zeros((n,m))
    weigths_output=np.zeros((m,k))
    gen=0
    for i in range(n):
        for j in range(m):
            weight_hiden[i][j]=cromosome[gen]
            gen+=1
    for i in range(m):
        for j in range(k):
            weigths_output[i][j] = cromosome[gen]
            gen += 1
    return  weight_hiden, weigths_output

def sort_popualtion(population,results):
    results1={}
    population1={}
    list_sort=[]
    for i in results:
        list_sort.append(results[i])
    list_sort.sort()
    for i in range(len(list_sort)):
        for j in results:
            if results[j]==list_sort[len(list_sort)-i-1]:
                results1[i]=results[j]
                population1[i]=population[j]
    return population1,results1

def crossover(population):
    len_crome=len(population[0])
    new_generation=[]
    for i in range(len(population)//2):
        point=random.randint(0,len_crome)
        new_one=[]
        for j in range(point):
            new_one.append(population[2*i][i])
        for j in range(point,len_crome):
            new_one.append(population[2*i+1][j])
        new_generation.append(new_one)
    for i in range(len(new_generation)):
        population[i+len(population)//2]=new_generation[i]
    return population

def mutation(population):
    for i in population:
        if not random.randint(0,20):
            point=random.randint(0,len(population[i])-5)
            for j in range(5):
                population[i][point+j]=random.random()

def fitnes_function(object_bot,turn,objects_in_game):
    size = (object_bot.size/person_size-1)*5
    dead_by_wall=-10
    if object_bot.dead:
        dead=-5
    else:
        dead=1
    leftt=object_bot.x
    if leftt<=object_bot.speed:
        leftt=dead_by_wall
        dead=0
    else:
        leftt=0.5

    rightt = width_win-object_bot.x-object_bot.size
    if rightt <= object_bot.speed:
        rightt= dead_by_wall
        dead=0
    else:
        rightt=0.5

    upp = object_bot.y
    if upp <= object_bot.speed:
        upp = dead_by_wall
        dead=0
    else:
        upp = 0.5

    downn = heigth_win-object_bot.y-object_bot.size
    if downn <= object_bot.speed:
        downn = dead_by_wall
        dead=0
    else:
        downn = 0.5

    life_time=0
    if object_bot.dead==0:
        life_time+=5
    life_time+=object_bot.life_turn/10
    znach=size+life_time+leftt+rightt+upp+downn+dead
    return znach

def create_neiro_structure(input_data,cromosome,object_bot,turn,objects_in_game):
    data=[input_data['UP'],
          input_data['LEFT'],
          input_data['DOWN'],
          input_data['RIGHT'],
          #(boot_shoot_delay-object_bot.turn_till_last_shoot)*object_bot.speed,
          max(object_bot.x,object_bot.y,(width_win-object_bot.x-object_bot.size),(heigth_win-object_bot.y-object_bot.size)),
          object_bot.x,
          object_bot.y,
          (width_win-object_bot.x-object_bot.size),
          (heigth_win-object_bot.y-object_bot.size),
          object_bot.size,
          #face_nap(object_bot.face),
          #object_bot.life_turn*object_bot.speed,
          1]
    weigths_hide,weigths_output=cromosome_to_weights(cromosome,n,m,k)
    hidden1=np.dot(data,weigths_hide)
    for i in range(len(hidden1)):
        hidden1[i]=sigmoid(hidden1[i])
    output=np.dot(hidden1,weigths_output)
    for i in range(len(output)):
        output[i]=sigmoid(output[i])
    maxx=0
    maxx_num=-1
    for i in range(len(output)):
        if maxx<=output[i] and (i!=4 or object_bot.turn_till_last_shoot>=25):
            maxx=output[i]
            maxx_num=i
    # result=fitnes_function(object_bot,turn,objects_in_game)
    return choose_comand(maxx_num)

def action_in_game(objects_in_game,turn,population,results):
    is_running=True
    keys = pygame.key.get_pressed()  # Получаем нажатые кнопки
    if len(objects_in_game['player'])!=0:
        objects_in_game['player'][0].x, objects_in_game['player'][0].y, objects_in_game['player'][0].face,objects_in_game['player'][0].dead = action_object(
            objects_in_game['player'][0].x,
            objects_in_game['player'][0].y,
            objects_in_game['player'][0].speed,
            objects_in_game['player'][0].size,
            objects_in_game['player'][0].face,
            key_interpritation(keys),
            objects_in_game['player'][0].name,
            objects_in_game['player'][0].dead,
            objects_in_game)
    # else:#Смерть Игрока
    #     is_running = False
    for i in objects_in_game['bots']:
        sensors_i=sensors(objects_in_game,i)
        comand_bot_i=create_neiro_structure(sensors_i,population[i.name],i,turn,objects_in_game)
        if comand_bot_i=='SHOOT':
            i.turn_till_last_shoot=0
        i.x,i.y,i.face,i.dead=action_object(i.x,i.y,i.speed,i.size,i.face,comand_bot_i,i.name,i.dead,objects_in_game)
        i.turn_till_last_shoot+=1
    if len(objects_in_game['bots'])<=0 or turn>=10000:
        is_running=False
    for i in range(len(objects_in_game['boolets'])):
        if not objects_in_game['boolets'][i].is_dead_f():
            objects_in_game['boolets'][i].x,objects_in_game['boolets'][i].y,objects_in_game['boolets'][i].face,objects_in_game['boolets'][i].dead,=action_object(objects_in_game['boolets'][i].x,
                                                                                                                             objects_in_game['boolets'][i].y,
                                                                                                                             objects_in_game['boolets'][i].speed,
                                                                                                                             objects_in_game['boolets'][i].size,
                                                                                                                             objects_in_game['boolets'][i].face,
                                                                                                                             objects_in_game['boolets'][i].face,
                                                                                                                             objects_in_game['boolets'][i].name,
                                                                                                                             objects_in_game['boolets'][i].dead,
                                                                                                                             objects_in_game)
    for i in objects_in_game['comets']:
        if not i.is_dead_f():
            i.x,i.y,i.dead = comet_movement(objects_in_game,i.x,i.y,i.speed,i.size,i.dead)
    return is_running

def draw_objects(objects_in_game):
    for i in objects_in_game['comets']:
        pygame.draw.rect(win,i.color,(i.x,i.y,i.size,i.size))
    for i in objects_in_game['bots']:
        pygame.draw.rect(win,i.color,(i.x,i.y,i.size,i.size))
    for i in objects_in_game['player']:
        pygame.draw.rect(win,i.color,(i.x,i.y,i.size,i.size))
    for i in objects_in_game['boolets']:
        pygame.draw.rect(win,i.color,(i.x,i.y,i.size,i.size))


turn=0
is_exit=False
for run in range(500):
    objects_in_game = {'comets': [],
                       'bots': [],
                       'player': [],
                       'boolets': []}
    is_running=True
    keys = []  # список нажатых кнопок
    person_size = person_defolt  # размеры игрокаs
    # objects_in_game['player'].append(game_object(start_point_x,start_point_y,person_size,speed_move,(255,0,0),'player',name='player'))
    spawn_bots(objects_in_game, number_of_bots)
    turn = 0
    while is_running:
        turn+=1
        #pygame.time.delay(turn_delay) #задержка между действиями
        win.fill((0,0,0)) #очистка поля
        is_comet_spawned=random.randint(0,500)
        if not is_comet_spawned:
            spawn_comet(objects_in_game)
        is_running=action_in_game(objects_in_game,turn,population,results)
        del_collision_objects(objects_in_game)
        del_dead_objects(objects_in_game,dead_bots)

        for event in pygame.event.get(): #реакция на закрытие окна
            if event.type == pygame.QUIT:
                is_running=False
                is_exit=True

        # draw_objects(objects_in_game) #рисуем игрока Игрок
        # # if not is_running:
        # #     win.blit(endgamebad_i, endgamebad_rect)
        # pygame.display.update() #Обновляме изображение
    for i in dead_bots:
        results[dead_bots[i].name]=fitnes_function(dead_bots[i],turn,objects_in_game)
    for i in objects_in_game['bots']:
        results[i.name]=fitnes_function(i,turn,objects_in_game)
    population,results=sort_popualtion(population,results)
    population=crossover(population)
    mutation(population)
    print(results[0], run)
    if is_exit:
        break

sleep(0.6)
pygame.quit()