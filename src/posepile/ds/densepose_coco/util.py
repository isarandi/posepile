import functools

import numpy as np
import scipy.spatial

from posepile.paths import DATA_ROOT
from posepile.util.matlabfile import load_mat


@functools.lru_cache()
def get_iuv_converter():
    """Returns a function that converts IUV (body part index + part-specific uv surface coords)
     representation to FBC representation (SMPL vertex ID triplet + barycentrics)."""

    ALP_UV = load_mat(f'{DATA_ROOT}/densepose/UV_data/UV_Processed.mat')
    FaceIndices = np.array(ALP_UV['All_FaceIndices']).squeeze()
    FacesDensePose = ALP_UV['All_Faces'] - 1
    U_norm = np.array(ALP_UV['All_U_norm']).squeeze()
    V_norm = np.array(ALP_UV['All_V_norm']).squeeze()
    All_vertices = np.array(ALP_UV['All_vertices'])

    def barycentric_coordinates_exists(P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        if np.dot(vCrossW, vCrossU) < 0:
            return False
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        if np.dot(uCrossW, uCrossV) < 0:
            return False
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        return (r <= 1) & (t <= 1) & (r + t <= 1)

    def barycentric_coordinates(P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        return 1 - (r + t), r, t

    def IUV2FBC(I_point, U_point, V_point):
        P = [U_point, V_point, 0]
        # Get the face indices of this body part (I is the label of part)
        FaceIndicesNow = np.where(FaceIndices == I_point)
        # Get the faces themselves, which are tuples of vertex indices
        FacesNow = FacesDensePose[FaceIndicesNow]

        # Now get the coordinates of the first vertex of each face in the UV map
        P_0 = np.vstack((
            U_norm[FacesNow][:, 0],
            V_norm[FacesNow][:, 0],
            np.zeros_like(U_norm[FacesNow][:, 0]))).transpose()
        # Now get the coordinates of the second vertex of each face in the UV map
        P_1 = np.vstack((
            U_norm[FacesNow][:, 1],
            V_norm[FacesNow][:, 1],
            np.zeros_like(U_norm[FacesNow][:, 1]))).transpose()
        P_2 = np.vstack((
            U_norm[FacesNow][:, 2],
            V_norm[FacesNow][:, 2],
            np.zeros_like(U_norm[FacesNow][:, 2]))).transpose()

        # Check if our uv point is inside any of the faces in the uv map
        for i, (P0, P1, P2) in enumerate(zip(P_0, P_1, P_2)):
            if barycentric_coordinates_exists(P0, P1, P2, P):
                return FaceIndicesNow[0][i], *barycentric_coordinates(P0, P1, P2, P)

        # If not, find the closest face and return the barycentric coordinates of the closest vertex
        D1 = scipy.spatial.distance.cdist(
            np.array([U_point, V_point])[np.newaxis, :], P_0[:, 0:2]).squeeze()
        D2 = scipy.spatial.distance.cdist(
            np.array([U_point, V_point])[np.newaxis, :], P_1[:, 0:2]).squeeze()
        D3 = scipy.spatial.distance.cdist(
            np.array([U_point, V_point])[np.newaxis, :], P_2[:, 0:2]).squeeze()
        minD1 = D1.min()
        minD2 = D2.min()
        minD3 = D3.min()
        if minD1 < minD2 and minD1 < minD3:
            return FaceIndicesNow[0][np.argmin(D1)], 1., 0., 0.
        elif minD2 < minD1 and minD2 < minD3:
            return FaceIndicesNow[0][np.argmin(D2)], 0., 1., 0.
        else:
            return FaceIndicesNow[0][np.argmin(D3)], 0., 0., 1.

    def fn(I, U, V):
        I = np.array(I)
        U = np.array(U)
        V = np.array(V)
        fbc = np.array([IUV2FBC(i, u, v) for i, u, v in zip(I, U, V)])
        face_indices_dp = fbc[:, 0].astype(np.int32)
        barycoords = fbc[:, 1:].astype(np.float32)
        faces = All_vertices[FacesDensePose[face_indices_dp]] - 1
        return faces, barycoords

    return fn
