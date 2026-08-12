// grad.cpp —— Green-Gauss 梯度重构
#include "grad.h"

#include <stdexcept>

namespace cc {

namespace {
// 取面 jacobian 第一行(法向向量) 作为 Vector2d
inline Eigen::Vector2d nvec(const face_class* f) {
    return Eigen::Vector2d(f->jacobian(0, 0), f->jacobian(0, 1));
}
}  // namespace

std::map<std::string, Eigen::Vector2d> green_gauss_face_vari(face_class* face) {
    std::map<std::string, Eigen::Vector2d> grad_dic;
    if (face->direction == "WE") {
        cell_class* east = face->east;
        cell_class* west = face->west;
        grad_dic["w"] = (nvec(west->north) - nvec(west->south) + nvec(west->east) - nvec(west->west)) / (west->vol * 4) -
                        nvec(east->west) / (east->vol * 4);
        grad_dic["e"] = (nvec(east->north) - nvec(east->south) + nvec(east->east) - nvec(east->west)) / (east->vol * 4) +
                        nvec(west->east) / (west->vol * 4);
        grad_dic["nw"] = nvec(west->north) / (west->vol * 4);
        grad_dic["ne"] = nvec(east->north) / (east->vol * 4);
        grad_dic["sw"] = -nvec(west->south) / (west->vol * 4);
        grad_dic["se"] = -nvec(east->south) / (east->vol * 4);
        grad_dic["ee"] = nvec(east->east) / (east->vol * 4);
        grad_dic["ww"] = -nvec(west->west) / (west->vol * 4);
        return grad_dic;
    } else if (face->direction == "NS") {
        cell_class* north = face->north;
        cell_class* south = face->south;
        grad_dic["n"] = (nvec(north->north) - nvec(north->south) + nvec(north->east) - nvec(north->west)) / (north->vol * 4) +
                        nvec(south->north) / (south->vol * 4);
        grad_dic["s"] = (nvec(south->north) - nvec(south->south) + nvec(south->east) - nvec(south->west)) / (south->vol * 4) -
                        nvec(north->south) / (north->vol * 4);
        grad_dic["ne"] = nvec(north->east) / (north->vol * 4);
        grad_dic["se"] = nvec(south->east) / (south->vol * 4);
        grad_dic["nw"] = -nvec(north->west) / (north->vol * 4);
        grad_dic["sw"] = -nvec(south->west) / (south->vol * 4);
        grad_dic["nn"] = nvec(north->north) / (north->vol * 4);
        grad_dic["ss"] = -nvec(south->south) / (south->vol * 4);
        return grad_dic;
    } else {
        throw std::runtime_error("face.direction must be 'WE' or 'NS'");
    }
}

std::map<std::string, Eigen::Vector2d> green_gauss_cell_vari(cell_class* cell) {
    std::map<std::string, Eigen::Vector2d> grad_dic;
    grad_dic["c"] = (nvec(cell->north) + nvec(cell->south) + nvec(cell->east) + nvec(cell->west)) / (cell->vol * 2);
    grad_dic["n"] = nvec(cell->north) / (cell->vol * 2);
    grad_dic["s"] = -nvec(cell->south) / (cell->vol * 2);
    grad_dic["e"] = nvec(cell->east) / (cell->vol * 2);
    grad_dic["w"] = -nvec(cell->west) / (cell->vol * 2);
    return grad_dic;
}

void green_gauss_from_JST(cell_class* cell, face_class* facenorth, face_class* facesouth,
                          face_class* faceeast, face_class* facewest) {
    Eigen::Vector4d u_vec(facenorth->u, facesouth->u, faceeast->u, facewest->u);
    Eigen::Vector4d v_vec(facenorth->v, facesouth->v, faceeast->v, facewest->v);
    Eigen::Vector4d miubl_vec(facenorth->miubl, facesouth->miubl, faceeast->miubl, facewest->miubl);
    Eigen::Vector4d T_vec(facenorth->T, facesouth->T, faceeast->T, facewest->T);
    Eigen::Vector4d nx_vec(facenorth->nx(), -facesouth->nx(), faceeast->nx(), -facewest->nx());
    Eigen::Vector4d ny_vec(facenorth->ny(), -facesouth->ny(), faceeast->ny(), -facewest->ny());
    cell->ugrad = Eigen::Vector2d(u_vec.dot(nx_vec), u_vec.dot(ny_vec)) / cell->vol;
    cell->vgrad = Eigen::Vector2d(v_vec.dot(nx_vec), v_vec.dot(ny_vec)) / cell->vol;
    cell->miublgrad = Eigen::Vector2d(miubl_vec.dot(nx_vec), miubl_vec.dot(ny_vec)) / cell->vol;
    cell->Tgrad = Eigen::Vector2d(T_vec.dot(nx_vec), T_vec.dot(ny_vec)) / cell->vol;
}

}  // namespace cc
