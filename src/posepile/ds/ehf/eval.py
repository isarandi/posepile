def evaluate(self, outs, cur_sample_idx):
    annots = self.datalist
    sample_num = len(outs)
    eval_result = {'pa_mpvpe_all': [], 'pa_mpvpe_l_hand': [], 'pa_mpvpe_r_hand': [],
                   'pa_mpvpe_hand': [], 'pa_mpvpe_face': [],
                   'mpvpe_all': [], 'mpvpe_l_hand': [], 'mpvpe_r_hand': [], 'mpvpe_hand': [],
                   'mpvpe_face': [],
                   'pa_mpjpe_body': [], 'pa_mpjpe_l_hand': [], 'pa_mpjpe_r_hand': [],
                   'pa_mpjpe_hand': []}

    for n in range(sample_num):
        annot = annots[cur_sample_idx + n]
        ann_id = annot['img_path'].split('/')[-1].split('_')[0]
        out = outs[n]

        # MPVPE from all vertices
        mesh_gt = np.dot(self.cam_param['R'],
                         out['smplx_mesh_cam_target'].transpose(1, 0)).transpose(1, 0)
        mesh_out = out['smplx_mesh_cam']

        mesh_out_align = rigid_align(mesh_out, mesh_gt)
        pa_mpvpe_all = np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000
        eval_result['pa_mpvpe_all'].append(pa_mpvpe_all)
        mesh_out_align = mesh_out - np.dot(smpl_x.J_regressor, mesh_out)[
                                    smpl_x.J_regressor_idx['pelvis'], None,
                                    :] + np.dot(smpl_x.J_regressor, mesh_gt)[
                                         smpl_x.J_regressor_idx['pelvis'], None,
                                         :]
        mpvpe_all = np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000
        eval_result['mpvpe_all'].append(mpvpe_all)

        # MPVPE from hand vertices
        mesh_gt_lhand = mesh_gt[smpl_x.hand_vertex_idx['left_hand'], :]
        mesh_out_lhand = mesh_out[smpl_x.hand_vertex_idx['left_hand'], :]
        mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
        mesh_gt_rhand = mesh_gt[smpl_x.hand_vertex_idx['right_hand'], :]
        mesh_out_rhand = mesh_out[smpl_x.hand_vertex_idx['right_hand'], :]
        mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)
        eval_result['pa_mpvpe_l_hand'].append(np.sqrt(
            np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000)
        eval_result['pa_mpvpe_r_hand'].append(np.sqrt(
            np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000)
        eval_result['pa_mpvpe_hand'].append((np.sqrt(
            np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
            np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

        mesh_out_lhand_align = mesh_out_lhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                                smpl_x.J_regressor_idx['lwrist'], None, :] + np.dot(
            smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['lwrist'], None, :]
        mesh_out_rhand_align = mesh_out_rhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                                smpl_x.J_regressor_idx['rwrist'], None, :] + np.dot(
            smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['rwrist'], None, :]

        eval_result['mpvpe_l_hand'].append(np.sqrt(
            np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000)
        eval_result['mpvpe_r_hand'].append(np.sqrt(
            np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000)
        eval_result['mpvpe_hand'].append((np.sqrt(
            np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
            np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

        # MPVPE from face vertices
        mesh_gt_face = mesh_gt[smpl_x.face_vertex_idx, :]
        mesh_out_face = mesh_out[smpl_x.face_vertex_idx, :]
        mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
        eval_result['pa_mpvpe_face'].append(
            np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)
        mesh_out_face_align = mesh_out_face - np.dot(smpl_x.J_regressor, mesh_out)[
                                              smpl_x.J_regressor_idx['neck'],
                                              None, :] + np.dot(smpl_x.J_regressor, mesh_gt)[
                                                         smpl_x.J_regressor_idx['neck'], None, :]
        eval_result['mpvpe_face'].append(
            np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)

        # MPJPE from body joints
        joint_gt_body = np.dot(smpl_x.j14_regressor, mesh_gt)
        joint_out_body = np.dot(smpl_x.j14_regressor, mesh_out)
        joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
        eval_result['pa_mpjpe_body'].append(
            np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, 1)).mean() * 1000)

        # MPJPE from hand joints
        joint_gt_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_gt)
        joint_out_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_out)
        joint_out_lhand_align = rigid_align(joint_out_lhand, joint_gt_lhand)
        joint_gt_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_gt)
        joint_out_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_out)
        joint_out_rhand_align = rigid_align(joint_out_rhand, joint_gt_rhand)
        eval_result['pa_mpjpe_l_hand'].append(np.sqrt(
            np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)).mean() * 1000)
        eval_result['pa_mpjpe_r_hand'].append(np.sqrt(
            np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)).mean() * 1000)
        eval_result['pa_mpjpe_hand'].append((np.sqrt(
            np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
            np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)


    print('======EHF======')
    print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
    print('PA MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_l_hand']))
    print('PA MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_r_hand']))
    print('PA MPVPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_hand']))
    print('PA MPVPE (Face): %.2f mm' % np.mean(eval_result['pa_mpvpe_face']))
    print()

    print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
    print('MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['mpvpe_l_hand']))
    print('MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['mpvpe_r_hand']))
    print('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
    print('MPVPE (Face): %.2f mm' % np.mean(eval_result['mpvpe_face']))
    print()

    print('PA MPJPE (Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body']))
    print('PA MPJPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_l_hand']))
    print('PA MPJPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_r_hand']))
    print('PA MPJPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_hand']))
    print()

    print(
        f"{np.mean(eval_result['pa_mpvpe_all'])},{np.mean(eval_result['pa_mpvpe_l_hand'])},{np.mean(eval_result['pa_mpvpe_r_hand'])},{np.mean(eval_result['pa_mpvpe_hand'])},{np.mean(eval_result['pa_mpvpe_face'])},"
        f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])}")
    print()