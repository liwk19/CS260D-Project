from glob import iglob
from pprint import pprint
from os import rename
import pickle5 as pickle

prev_name = 'fdtd_'
new_name = 'fdtd-2d_'
path = './poly/databases/**/*'

obj = 'tst'
pickled_obj = pickle.dumps(obj)
print(pickled_obj)
pickled_obj = b'\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04test\x94.'.replace(b'\x04test', b'\x03tst')
print(pickled_obj)
# obj = pickle.loads(pickled_obj)

p=b'\x80\x04\x95\xa4\x19\x00\x00\x00\x00\x00\x00\x8c\x0eautudse.result\x94\x8c\x0cMerlinResult\x94\x93\x94)\x81\x94}\x94(\x8c\x05point\x94}\x94(\x8c\n__PARA__L0\x94K\x01\x8c\n__PIPE__L0\x94\x8c\x07flatten\x94\x8c\n__TILE__L0\x94K\x01u\x8c\x08ret_code\x94h\x00\x8c\x0eResult.RetCode\x94\x93\x94K\x00\x85\x94R\x94\x8c\x05valid\x94\x88\x8c\x04path\x94N\x8c\x07quality\x94G\xff\xf0\x00\x00\x00\x00\x00\x00\x8c\x04perf\x94G\x00\x00\x00\x00\x00\x00\x00\x00\x8c\x08res_util\x94}\x94(\x8c\tutil-BRAM\x94K\x00\x8c\x08util-DSP\x94K\x00\x8c\x08util-LUT\x94K\x00\x8c\x07util-FF\x94K\x00\x8c\ntotal-BRAM\x94K\x00\x8c\ttotal-DSP\x94K\x00\x8c\ttotal-LUT\x94K\x00\x8c\x08total-FF\x94K\x00u\x8c\teval_time\x94G\x00\x00\x00\x00\x00\x00\x00\x00\x8c\tcriticals\x94]\x94\x8c\tcode_hash\x94X.\x18\x00\x00#define__constant#define__kernel#define__global#include"memcpy_512_1d.h"#defineSIZE_116#include"memcpy_512_2d.h"#undefSIZE_1#include<string.h>#include"merlin_type_define.h"staticvoid__merlin_dummy_extern_int_merlin_include_G_();staticvoid__merlin_dummy_extern_int_merlin_include_G_();staticvoid__merlin_dummy_extern_int_merlin_include_G_();staticvoid__merlin_dummy_extern_int_merlin_include_G_();staticvoid__merlin_dummy_kernel_pragma();/*Original:#pragmaACCELkernel*/voidmd_kernel(merlin_uint_512force_x[32],merlin_uint_512force_y[32],merlin_uint_512force_z[32],merlin_uint_512position_x[32],merlin_uint_512position_y[32],merlin_uint_512position_z[32],merlin_uint_512NL[256]){#pragmaHLSINTERFACEm_axiport=NLoffset=slavedepth=256bundle=merlin_gmem_md_kernel_512_0#pragmaHLSINTERFACEm_axiport=force_xoffset=slavedepth=32bundle=merlin_gmem_md_kernel_512_0#pragmaHLSINTERFACEm_axiport=force_yoffset=slavedepth=32bundle=merlin_gmem_md_kernel_512_1#pragmaHLSINTERFACEm_axiport=force_zoffset=slavedepth=32bundle=merlin_gmem_md_kernel_512_2#pragmaHLSINTERFACEm_axiport=position_xoffset=slavedepth=32bundle=merlin_gmem_md_kernel_512_position_x#pragmaHLSINTERFACEm_axiport=position_yoffset=slavedepth=32bundle=merlin_gmem_md_kernel_512_position_y#pragmaHLSINTERFACEm_axiport=position_zoffset=slavedepth=32bundle=merlin_gmem_md_kernel_512_position_z#pragmaHLSINTERFACEs_axiliteport=NLbundle=control#pragmaHLSINTERFACEs_axiliteport=force_xbundle=control#pragmaHLSINTERFACEs_axiliteport=force_ybundle=control#pragmaHLSINTERFACEs_axiliteport=force_zbundle=control#pragmaHLSINTERFACEs_axiliteport=position_xbundle=control#pragmaHLSINTERFACEs_axiliteport=position_ybundle=control#pragmaHLSINTERFACEs_axiliteport=position_zbundle=control#pragmaHLSINTERFACEs_axiliteport=returnbundle=control#pragmaHLSDATA_PACKVARIABLE=NL#pragmaHLSDATA_PACKVARIABLE=position_z#pragmaHLSDATA_PACKVARIABLE=position_y#pragmaHLSDATA_PACKVARIABLE=position_x#pragmaHLSDATA_PACKVARIABLE=force_z#pragmaHLSDATA_PACKVARIABLE=force_y#pragmaHLSDATA_PACKVARIABLE=force_x#pragmaACCELinterfacevariable=NLdepth=4096max_depth=4096#pragmaACCELinterfacevariable=position_zdepth=256max_depth=256#pragmaACCELinterfacevariable=position_ydepth=256max_depth=256#pragmaACCELinterfacevariable=position_xdepth=256max_depth=256#pragmaACCELinterfacevariable=force_zdepth=256max_depth=256#pragmaACCELinterfacevariable=force_ydepth=256max_depth=256#pragmaACCELinterfacevariable=force_xdepth=256max_depth=256doubleposition_z_2_0_buf[256];#pragmaHLSarray_partitionvariable=position_z_2_0_bufcyclicfactor=8dim=1doubleposition_y_2_0_buf[256];#pragmaHLSarray_partitionvariable=position_y_2_0_bufcyclicfactor=8dim=1doubleposition_x_2_0_buf[256];#pragmaHLSarray_partitionvariable=position_x_2_0_bufcyclicfactor=8dim=1intNL_3_0_buf[256][16];#pragmaHLSarray_partitionvariable=NL_3_0_bufcompletedim=2doubleforce_z_buf[256];#pragmaHLSarray_partitionvariable=force_z_bufcyclicfactor=8dim=1doubleforce_y_buf[256];#pragmaHLSarray_partitionvariable=force_y_bufcyclicfactor=8dim=1doubleforce_x_buf[256];#pragmaHLSarray_partitionvariable=force_x_bufcyclicfactor=8dim=1doubledelx;doubledely;doubledelz;doubler2inv;doubler6inv;doublepotential;doubleforce;doublej_x;doublej_y;doublej_z;doublei_x;doublei_y;doublei_z;doublefx;doublefy;doublefz;inti;intj;intjidx;{memcpy_wide_bus_read_int_2d_16_512(NL_3_0_buf,0,0,(merlin_uint_512*)NL,(0*4),sizeof(int)*((unsignedlong)4096),4096L);}{memcpy_wide_bus_read_double_512(&position_x_2_0_buf[0],(merlin_uint_512*)position_x,(0*8),sizeof(double)*((unsignedlong)256),256L);}{memcpy_wide_bus_read_double_512(&position_y_2_0_buf[0],(merlin_uint_512*)position_y,(0*8),sizeof(double)*((unsignedlong)256),256L);}{memcpy_wide_bus_read_double_512(&position_z_2_0_buf[0],(merlin_uint_512*)position_z,(0*8),sizeof(double)*((unsignedlong)256),256L);}loopui0:{merlinL0:for(i=0;i<256;i++)/*Original:#pragmaACCELPIPELINEflatten*//*Original:#pragmaACCELTILEFACTOR=1*//*Original:#pragmaACCELPARALLELFACTOR=1*//*Original:#pragmaACCELPIPELINE*//*Original:#pragmaACCELTILEFACTOR=1*//*Original:#pragmaACCELPARALLELFACTOR=1*//*Original:#pragmaACCELPIPELINE*//*Original:#pragmaACCELTILEFACTOR=1*/{#pragmaHLSdependencevariable=force_x_bufarrayinterfalse#pragmaHLSdependencevariable=force_y_bufarrayinterfalse#pragmaHLSdependencevariable=force_z_bufarrayinterfalse#pragmaHLSpipelinei_x=position_x_2_0_buf[i];i_y=position_y_2_0_buf[i];i_z=position_z_2_0_buf[i];fx=((double)0);fy=((double)0);fz=((double)0);loopuj0:for(j=0;j<16;j++)/*Original:#pragmaACCELPARALLELCOMPLETE*//*Original:#pragmaACCELPARALLELCOMPLETE*/{#pragmaHLSunrolldoubletmp_1;doubletmp_0;doubletmp;jidx=NL_3_0_buf[i][j];tmp=memcpy_wide_bus_single_read_double_512((merlin_uint_512*)position_x,(size_t)(jidx*8));j_x=tmp;tmp_0=memcpy_wide_bus_single_read_double_512((merlin_uint_512*)position_y,(size_t)(jidx*8));j_y=tmp_0;tmp_1=memcpy_wide_bus_single_read_double_512((merlin_uint_512*)position_z,(size_t)(jidx*8));j_z=tmp_1;delx=i_x-j_x;dely=i_y-j_y;delz=i_z-j_z;r2inv=1.0/(delx*delx+dely*dely+delz*delz);r6inv=r2inv*r2inv*r2inv;potential=r6inv*(1.5*r6inv-2.0);force=r2inv*potential;fx+=delx*force;fy+=dely*force;fz+=delz*force;}force_x_buf[i]=fx;force_y_buf[i]=fy;force_z_buf[i]=fz;}/*ExistingHLSpartition:#pragmaHLSarray_partitionvariable=force_z_bufcyclicfactor=8dim=1*/memcpy_wide_bus_write_double_512((merlin_uint_512*)force_z,&force_z_buf[0],(8*0),sizeof(double)*((unsignedlong)256),256L);/*ExistingHLSpartition:#pragmaHLSarray_partitionvariable=force_y_bufcyclicfactor=8dim=1*/memcpy_wide_bus_write_double_512((merlin_uint_512*)force_y,&force_y_buf[0],(8*0),sizeof(double)*((unsignedlong)256),256L);/*ExistingHLSpartition:#pragmaHLSarray_partitionvariable=force_x_bufcyclicfactor=8dim=1*/memcpy_wide_bus_write_double_512((merlin_uint_512*)force_x,&force_x_buf[0],(8*0),sizeof(double)*((unsignedlong)256),256L);}}/*ExistingHLSpartition:#pragmaHLSarray_partitionvariable=NL_3_0_bufcyclicfactor=16dim=2*//*ExistingHLSpartition:#pragmaHLSarray_partitionvariable=position_x_2_0_bufcyclicfactor=8dim=1*//*ExistingHLSpartition:#pragmaHLSarray_partitionvariable=position_y_2_0_bufcyclicfactor=8dim=1*//*ExistingHLSpartition:#pragmaHLSarray_partitionvariable=position_z_2_0_bufcyclicfactor=8dim=1*/\x94ub.'
print(p.replace(b'autudse', b'autodse'))
# files = [f for f in iglob(path, recursive=True) if f.endswith('.db') and prev_name in f]
# pprint(files)
# for f in files:
#     rename(f, f.replace(prev_name, new_name))