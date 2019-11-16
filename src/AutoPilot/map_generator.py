from base_xml import get_base_xml
import random
random.seed(1)
class Cfg:
    start_x,start_y,start_z = (1,2,-3)
    # end_x,end_y,end_z = (-4,2,30)
    maze_bound = (-4,-5,4,25)
    number_of_walls = 5
    end_block_type = "redstone_block"
    road_block_type = "diamond_block"
    wall_block_type = "iron_block"
    seed = 0



def get_maze_xml():
    cfg = Cfg()
    road_str = f'<DrawCuboid x1="{cfg.maze_bound[0]}"  y1="1" z1="{cfg.maze_bound[1]}"  x2="{cfg.maze_bound[2]}" y2="-1" z2="{cfg.maze_bound[3]}" type="{cfg.road_block_type}" /> '
    destination = f'<DrawCuboid x1="{cfg.maze_bound[0]}"  y1="-3" z1="{cfg.maze_bound[3]}"  x2="{cfg.maze_bound[2]}" y2="15" z2="{cfg.maze_bound[3]}" type="{cfg.end_block_type}" /> \n'

    out_wall = f'<DrawCuboid x1="{cfg.maze_bound[0]-2}"  y1="3" z1="{cfg.maze_bound[1]-2}"  x2="{cfg.maze_bound[2]+2}" y2="1" z2="{cfg.maze_bound[3]+2}" type="{cfg.wall_block_type}" /> \n' \
               f'<DrawCuboid x1="{cfg.maze_bound[0]-1}"  y1="3" z1="{cfg.maze_bound[1]-1}"  x2="{cfg.maze_bound[2]+1}" y2="2" z2="{cfg.maze_bound[3]+1}" type="air" />'
    #out_wall = ''
    block_list = ""
    for i in range(cfg.number_of_walls):
        while True:
            x,z = random.randint(cfg.maze_bound[0],cfg.maze_bound[2]),random.randint(cfg.maze_bound[1],cfg.maze_bound[3])
            if abs(cfg.start_x - x) <=3 and abs(cfg.start_z - z) <= 3:
                continue
            break
        block_list += f'<DrawCuboid x1="{x}"  y1="0" z1="{z}"  x2="{x}" y2="3" z2="{z}" type="{cfg.wall_block_type}" /> \n '

    block_list += f'<DrawCuboid x1="{cfg.start_x}"  y1="0" z1="{cfg.start_z+4}"  x2="{cfg.start_x}" y2="3" z2="{cfg.start_z+4}" type="{cfg.wall_block_type}" /> \n '
    block_list += f'<DrawCuboid x1="{cfg.start_x-2}"  y1="0" z1="{cfg.start_z + 10}"  x2="{cfg.start_x-2}" y2="3" z2="{cfg.start_z + 10}" type="{cfg.wall_block_type}" /> \n '
    #cfg.start_x,cfg.start_z = random.randint(cfg.maze_bound[0]+1,cfg.maze_bound[2]-1),random.randint(cfg.maze_bound[1]+1,cfg.maze_bound[3]-1)
    return get_base_xml(out_wall,road_str,destination,block_list,cfg.start_x,cfg.start_y,cfg.start_z,cfg.maze_bound)

def save_maze_xml(num=20):
    for n in range(num):
        cfg = Cfg()
        road_str = f'<DrawCuboid x1="{cfg.maze_bound[0]}"  y1="1" z1="{cfg.maze_bound[1]}"  x2="{cfg.maze_bound[2]}" y2="-1" z2="{cfg.maze_bound[3]}" type="{cfg.road_block_type}" /> '
        destination = f'<DrawCuboid x1="{cfg.maze_bound[0]}"  y1="-3" z1="{cfg.maze_bound[3]}"  x2="{cfg.maze_bound[2]}" y2="15" z2="{cfg.maze_bound[3]}" type="{cfg.end_block_type}" /> \n'

        out_wall = f'<DrawCuboid x1="{cfg.maze_bound[0] - 2}"  y1="3" z1="{cfg.maze_bound[1] - 2}"  x2="{cfg.maze_bound[2] + 2}" y2="1" z2="{cfg.maze_bound[3] + 2}" type="{cfg.wall_block_type}" /> \n' \
                   f'<DrawCuboid x1="{cfg.maze_bound[0] - 1}"  y1="3" z1="{cfg.maze_bound[1] - 1}"  x2="{cfg.maze_bound[2] + 1}" y2="2" z2="{cfg.maze_bound[3] + 1}" type="air" />'
        # out_wall = ''
        block_list = ""
        for i in range(cfg.number_of_walls):
            while True:
                x, z = random.randint(cfg.maze_bound[0], cfg.maze_bound[2]), random.randint(cfg.maze_bound[1],
                                                                                            cfg.maze_bound[3])
                if abs(cfg.start_x - x) <= 3 and abs(cfg.start_z - z) <= 3:
                    continue
                break
            block_list += f'<DrawCuboid x1="{x}"  y1="0" z1="{z}"  x2="{x}" y2="3" z2="{z}" type="{cfg.wall_block_type}" /> \n '

        block_list += f'<DrawCuboid x1="{cfg.start_x}"  y1="0" z1="{cfg.start_z + 3}"  x2="{cfg.start_x}" y2="3" z2="{cfg.start_z + 3}" type="{cfg.wall_block_type}" /> \n '
        cfg.start_x, cfg.start_z = random.randint(cfg.maze_bound[0] + 1, cfg.maze_bound[2] - 1), random.randint(
            cfg.maze_bound[1] + 1, cfg.maze_bound[3] - 1)
        xml_str = get_base_xml(out_wall, road_str, destination, block_list, cfg.start_x, cfg.start_y, cfg.start_z,
                            cfg.maze_bound)
        with open(f"xmls/world_{n}.txt", "w") as f:
            f.write(xml_str)
def load_maze_xml(i):
    xml_str = ""
    with open(f"xmls/world_{i}.txt", "r") as f:
        xml_str = f.read()
    return xml_str


if __name__ == "__main__":
    save_maze_xml(450)
