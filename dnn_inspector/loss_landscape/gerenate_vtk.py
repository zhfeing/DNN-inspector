"""
    Convert pth files to vtp files in VTK XML format that can be opened by ParaView.
    The data type of the vtp file == "vtkPolyData", each PolyData piece specifies a set
    of points and cells independently from the other pieces. The points are described
    explicitly by the Points element. The cells are described explicitly by the Verts,
    Lines, Strips, and Polys elements.

    <VTKFile type="PolyData" ...>
        <PolyData>
            <Piece NumberOfPoints="#" NumberOfVerts="#" NumberOfLines="#"
            NumberOfStrips="#" NumberOfPolys="#">
                <PointData>...</PointData>
                <CellData>...</CellData>
                <Points>...</Points>
                <Verts>...</Verts>
                <Lines>...</Lines>
                <Strips>...</Strips>
                <Polys>...</Polys>
            </Piece>
        </PolyData>
    </VTKFile>
"""
import logging
import math
from typing import Dict

import numpy as np
from scipy import interpolate

import torch


def prepare_data(
    eval_fp: str,
    coordinates_fp: str,
    z_err_max: float = -1,
    normalize_loss: float = -1,
    z_loss_max: float = -1,
    num_interpolate_points: int = -1,
):
    coordinates: Dict[str, torch.Tensor] = torch.load(coordinates_fp, map_location="cpu")
    x_coordinate: np.ndarray = coordinates["x_coordinate"].numpy()
    y_coordinate: np.ndarray = coordinates["y_coordinate"].numpy()

    eval_ckpt: Dict[str, torch.Tensor] = torch.load(eval_fp, map_location="cpu")
    z_loss: np.ndarray = eval_ckpt["loss"].numpy()
    z_err: np.ndarray = 1 - eval_ckpt["acc"].numpy()

    assert len(x_coordinate) > 1 and len(y_coordinate) > 1, "x or y coordinates must more than one value"

    def prepare_data(z: np.ndarray, z_max: float = -1, log_space: bool = False, normalize: float = -1):
        if z_max > 0:
            z = np.clip(z, a_min=None, a_max=z_max)
        if log_space:
            z = np.log10(z + 0.1)
        if normalize > 0:
            z = z / z.max() * normalize

        # Interpolate the resolution up to the desired amount
        if num_interpolate_points > 0:
            z = interpolate.interp2d(x_coordinate, y_coordinate, z, kind='cubic')
            x_axis = np.linspace(x_coordinate.min(), x_coordinate.max(), num_interpolate_points)
            y_axis = np.linspace(y_coordinate.min(), y_coordinate.max(), num_interpolate_points)
            z = z(x_axis, y_axis).ravel()

            x_axis, y_axis = np.meshgrid(x_axis, y_axis)
        else:
            x_axis, y_axis = np.meshgrid(x_coordinate, y_coordinate)
            z = z.ravel()
        x_axis: np.ndarray = x_axis.ravel()
        y_axis: np.ndarray = y_axis.ravel()
        return x_axis, y_axis, z

    x_axis, y_axis, z_err = prepare_data(
        z=z_err,
        z_max=z_err_max,
        log_space=False
    )
    err_data = {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z": z_err
    }

    x_axis, y_axis, z_loss = prepare_data(
        z=z_loss,
        z_max=z_loss_max,
        log_space=True,
        normalize=normalize_loss
    )
    loss_data = {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z": z_loss
    }
    return err_data, loss_data


def convert_vtk(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z: np.ndarray,
    vtp_fp: str,
    show_points: bool = False,
    show_polys: bool = True,
):
    logger = logging.getLogger("convert_vtp")

    number_points = len(z)
    logger.info("surface has %d points", number_points)

    matrix_size = int(math.sqrt(number_points))
    logger.info("Matrix_size = {} x {}".format(matrix_size, matrix_size))

    poly_size = matrix_size - 1
    logger.info("Poly_size = {} x {}".format(poly_size, poly_size))

    number_polys = poly_size * poly_size
    logger.info("number_polys = {}".format(number_polys))

    min_value_array = [min(x_axis), min(y_axis), min(z)]
    max_value_array = [max(x_axis), max(y_axis), max(z)]
    min_value = min(min_value_array)
    max_value = max(max_value_array)

    averaged_z_value_array = []

    poly_count = 0
    for column_count in range(poly_size):
        stride_value = column_count * matrix_size
        for row_count in range(poly_size):
            temp_index = stride_value + row_count
            averaged_z_value = (z[temp_index] + z[temp_index + 1] + z[temp_index + matrix_size] + z[temp_index + matrix_size + 1]) / 4.0
            averaged_z_value_array.append(averaged_z_value)
            poly_count += 1

    avg_min_value = min(averaged_z_value_array)
    avg_max_value = max(averaged_z_value_array)

    output_file = open(vtp_fp, 'w')
    output_file.write('<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
    output_file.write('\t<PolyData>\n')

    if (show_points and show_polys):
        output_file.write('\t\t<Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(number_points, number_points, number_polys))
    elif (show_polys):
        output_file.write('\t\t<Piece NumberOfPoints="{}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(number_points, number_polys))
    else:
        output_file.write('\t\t<Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="">\n'.format(number_points, number_points))

    # <PointData>
    output_file.write('\t\t\t<PointData>\n')
    output_file.write('\t\t\t<DataArray type="Float32" Name="zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(min_value_array[2], max_value_array[2]))
    for vertexcount in range(number_points):
        if (vertexcount % 6) == 0:
            output_file.write('\t\t\t\t')
        output_file.write('{}'.format(z[vertexcount]))
        if (vertexcount % 6) == 5:
            output_file.write('\n')
        else:
            output_file.write('\t')
    if (vertexcount % 6) != 5:
        output_file.write('\n')
    output_file.write('\t\t\t</DataArray>\n')
    output_file.write('\t\t\t</PointData>\n')

    # <CellData>
    output_file.write('\t\t\t<CellData>\n')
    if (show_polys and not show_points):
        output_file.write('\t\t\t<DataArray type="Float32" Name="averaged zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(avg_min_value, avg_max_value))
        for vertexcount in range(number_polys):
            if (vertexcount % 6) == 0:
                output_file.write('\t\t\t\t')
            output_file.write('{}'.format(averaged_z_value_array[vertexcount]))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write('\t')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
        output_file.write('\t\t\t</DataArray>\n')
    output_file.write('\t\t\t</CellData>\n')

    # <Points>
    output_file.write('\t\t\t<Points>\n')
    output_file.write('\t\t\t<DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(min_value, max_value))
    for vertexcount in range(number_points):
        if (vertexcount % 2) == 0:
            output_file.write('\t\t\t\t')
        output_file.write('{} {} {}'.format(x_axis[vertexcount], y_axis[vertexcount], z[vertexcount]))
        if (vertexcount % 2) == 1:
            output_file.write('\n')
        else:
            output_file.write('\t')
    if (vertexcount % 2) != 1:
        output_file.write('\n')
    output_file.write('\t\t\t</DataArray>\n')
    output_file.write('\t\t\t</Points>\n')

    # <Verts>
    output_file.write('\t\t\t<Verts>\n')
    output_file.write('\t\t\t<DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_points - 1))
    if (show_points):
        for vertexcount in range(number_points):
            if (vertexcount % 6) == 0:
                output_file.write('\t\t\t\t')
            output_file.write('{}'.format(vertexcount))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write('\t')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
    output_file.write('\t\t\t</DataArray>\n')
    output_file.write('\t\t\t<DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_points))
    if (show_points):
        for vertexcount in range(number_points):
            if (vertexcount % 6) == 0:
                output_file.write('\t\t\t\t')
            output_file.write('{}'.format(vertexcount + 1))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write('\t')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
    output_file.write('\t\t\t</DataArray>\n')
    output_file.write('\t\t\t</Verts>\n')

    # <Lines>
    output_file.write('\t\t\t<Lines>\n')
    output_file.write('\t\t\t<DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    output_file.write('\t\t\t</DataArray>\n')
    output_file.write('\t\t\t<DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    output_file.write('\t\t\t</DataArray>\n')
    output_file.write('\t\t\t</Lines>\n')

    # <Strips>
    output_file.write('\t\t\t<Strips>\n')
    output_file.write('\t\t\t<DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    output_file.write('\t\t\t</DataArray>\n')
    output_file.write('\t\t\t<DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    output_file.write('\t\t\t</DataArray>\n')
    output_file.write('\t\t\t</Strips>\n')

    # <Polys>
    output_file.write('\t\t\t<Polys>\n')
    output_file.write('\t\t\t<DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    if (show_polys):
        polycount = 0
        for column_count in range(poly_size):
            stride_value = column_count * matrix_size
            for row_count in range(poly_size):
                temp_index = stride_value + row_count
                if (polycount % 2) == 0:
                    output_file.write('\t\t\t\t')
                output_file.write('{} {} {} {}'.format(temp_index, (temp_index + 1), (temp_index + matrix_size + 1), (temp_index + matrix_size)))
                if (polycount % 2) == 1:
                    output_file.write('\n')
                else:
                    output_file.write('\t')
                polycount += 1
        if (polycount % 2) == 1:
            output_file.write('\n')
    output_file.write('\t\t\t</DataArray>\n')
    output_file.write('\t\t\t<DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    if (show_polys):
        for polycount in range(number_polys):
            if (polycount % 6) == 0:
                output_file.write('\t\t\t\t')
            output_file.write('{}'.format((polycount + 1) * 4))
            if (polycount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write('\t')
        if (polycount % 6) != 5:
            output_file.write('\n')
    output_file.write('\t\t\t</DataArray>\n')
    output_file.write('\t\t\t</Polys>\n')

    output_file.write('\t\t</Piece>\n')
    output_file.write('\t</PolyData>\n')
    output_file.write('</VTKFile>\n')
    output_file.write('')
    output_file.close()

    logger.info("Done")


def generate_vtp(
    eval_fp: str,
    coordinates_fp: str,
    save_fp: str,
    z_err_max: float = -1,
    z_loss_max: float = 1.0e8,
    normalize_loss: float = 1.0,
    num_interpolate_points: int = 100,
    show_points: bool = True,
    show_polys: bool = True
):
    err_data, loss_data = prepare_data(
        eval_fp=eval_fp,
        coordinates_fp=coordinates_fp,
        z_err_max=z_err_max,
        z_loss_max=z_loss_max,
        normalize_loss=normalize_loss,
        num_interpolate_points=num_interpolate_points
    )
    convert_vtk(**err_data, vtp_fp=save_fp.format(name="err"), show_points=show_points, show_polys=show_polys)
    convert_vtk(**loss_data, vtp_fp=save_fp.format(name="loss"), show_points=show_points, show_polys=show_polys)

