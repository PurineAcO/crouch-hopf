import classconfig as cc
import numpy as np

def convect_mat(cell:cc.cell_class):
    # TODO 兴许重建前4*4就行了......
    cell.convect_x = [[cell.u,cell.rho,0,0,0],
                      [cell.u**2+ cc.R*cell.T,2*cell.rho*cell.u,0,cell.rho*cc.R,0],
                      [cell.u*cell.v,cell.rho*cell.v,cell.rho*cell.u,0,0],
                      [cell.u*cell.H,cell.rho*(cell.H+cell.u**2),cell.rho*cell.u*cell.v,cell.rho*cell.u*cc.cp,0],
                      [cell.u*cell.miubl,cell.rho*cell.miunl,0,0,cell.rho*cell.u]]
    cell.convect_y = [[cell.v,0,cell.rho,0,0],
                      [cell.u*cell.v,cell.rho*cell.v,cell.rho*cell.u,0,0],
                      [cell.v**2+ cc.R*cell.T,0,2*cell.rho*cell.v,cell.rho*cc.R,0],
                      [cell.v*cell.H,cell.rho*cell.u*cell.v,cell.rho*(cell.H+cell.v**2),cell.rho*cell.v*cc.cp,0],
                      [cell.v*cell.miubl,0,cell.rho*cell.miunl,0,cell.rho*cell.v]]