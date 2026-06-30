import torch

# -----------------------------
# Helper Autodiff util
# -----------------------------
def grad(outputs, inputs):
    # outputs: (N,1) or (N,)
    # inputs: (N,1)
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

def laplacian(f, x, y):
    fx = grad(f, x)
    fy = grad(f, y)
    fxx = grad(fx, x)
    fyy = grad(fy, y)
    return fxx + fyy

# helper: pegar colunas específicas do batch (já normalizado) e voltar pra escala original
def unnormalize_x(Xn, x_sd, x_mu):
    return Xn * x_sd + x_mu


def transform_nut(nut_log, mode="exp"):
    if mode == "exp":
        return torch.exp(nut_log)
    if mode == "exp10":
        return torch.pow(10.0, nut_log)
    if mode in ("identity", None):
        return nut_log
    raise ValueError(f"Unsupported nut transform: {mode}")

# Definindo os Resíduos físicos (2D incompressível, estacionário)
# ---------------------------------------------------------------------------------
# v1) Continuidade
# ---------------------------------------------------------------------------------
def pde_residuals_continuity(x, y, u_coarse, v_coarse, model_out):
    """
    x, y: (N,1) leaf tensors com requires_grad=True (os mesmos do forward)
    model_out:
      se predict_dp=True: (N,3) = [dU, dV, dP]
      senão: (N,2) = [dU, dV]
    """

    dU, dV = model_out[:, [0]], model_out[:, [1]]

    u_hat = u_coarse + dU
    v_hat = v_coarse + dV

    # derivadas
    u_x = grad(u_hat, x)
    v_y = grad(v_hat, y)

    r_cont = u_x + v_y

    return r_cont


def train_pinn_epoch_continuity(
    net,
    dl_data,
    opt,
    feat_index,
    x_mu,
    x_sd,
    y_mu,
    y_sd,
    w_data,
    w_cont,
    device,
    **unused,
):
    net.train()

    total_loss = 0.0
    total_data = 0.0
    total_cont = 0.0
    n_train = 0

    for Xn, Yn in dl_data:
        Xn = Xn.to(device)
        Yn = Yn.to(device)

        # Loss supervisionada em escala normalizada
        pred_data = net(Xn)
        loss_data = torch.mean((pred_data - Yn) ** 2)

        # Recupera X físico
        X_phys = unnormalize_x(Xn, x_sd, x_mu).detach()

        # x,y com gradiente para autograd
        x = X_phys[:, [feat_index["x"]]].clone().requires_grad_(True)
        y = X_phys[:, [feat_index["y"]]].clone().requires_grad_(True)

        # Reinjeta x,y no input físico
        X_phys_mod = X_phys.clone()
        X_phys_mod[:, [feat_index["x"]]] = x
        X_phys_mod[:, [feat_index["y"]]] = y

        # Renormaliza input
        Xn_mod = (X_phys_mod - x_mu) / x_sd

        # Predição normalizada e desnormalizada
        pred_phys_n = net(Xn_mod)
        pred_phys = pred_phys_n * y_sd + y_mu

        # Campos coarse físicos
        u_c = X_phys[:, [feat_index["Ux"]]]
        v_c = X_phys[:, [feat_index["Uy"]]]

        r_cont = pde_residuals_continuity(
            x=x,
            y=y,
            u_coarse=u_c,
            v_coarse=v_c,
            model_out=pred_phys
        )

        loss_cont = torch.mean(r_cont ** 2)

        loss = w_data * loss_data + w_cont * loss_cont

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bs = Xn.size(0)
        total_loss += loss.item() * bs
        total_data += loss_data.item() * bs
        total_cont += loss_cont.item() * bs
        n_train += bs

    return (total_loss / n_train, 
            total_data / n_train, 
            total_cont / n_train
    )

# -----------------------------
# v2) Continuidade + momento
# -----------------------------
def pde_residuals_cont_mom(
    x,
    y,
    u_coarse,
    v_coarse,
    p_coarse,
    nut_coarse,
    Re,
    model_out,
    nut_transform="log10",
):
    """
    x, y: (N,1) leaf tensors com requires_grad=True (os mesmos do forward)
    model_out:
      se predict_dp=True: (N,3) = [dU, dV, dP]
      senão: (N,2) = [dU, dV]
    """
    
    dU, dV, dP = model_out[:, [0]], model_out[:, [1]], model_out[:, [2]]

    p_hat = p_coarse + dP
    u_hat = u_coarse + dU
    v_hat = v_coarse + dV

    nu = 1.0 / (Re + 1e-12)
    nut_phys = transform_nut(nut_coarse, mode=nut_transform)
    nu_eff = nu + nut_phys

    # derivadas
    u_x = grad(u_hat, x); u_y = grad(u_hat, y)
    v_x = grad(v_hat, x); v_y = grad(v_hat, y)

    p_x = grad(p_hat, x); p_y = grad(p_hat, y)

    # laplacianos
    u_lap = laplacian(u_hat, x, y)
    v_lap = laplacian(v_hat, x, y)

    r_cont = u_x + v_y

    adv_u = u_hat * u_x + v_hat * u_y
    adv_v = u_hat * v_x + v_hat * v_y

    r_mom_u = adv_u + p_x - nu_eff* u_lap
    r_mom_v = adv_v + p_y - nu_eff* v_lap

    return r_cont, r_mom_u, r_mom_v

def train_pinn_epoch_cont_mom(
    net,
    dl_data,
    opt,
    feat_index,
    x_mu,
    x_sd,
    y_mu,
    y_sd,
    w_data,
    w_cont,
    w_mom,
    device,
    nut_transform="exp",
    **unused,
):
    net.train()

    total_loss = 0.0
    total_data = 0.0
    total_cont = 0.0
    total_mom = 0.0
    n_train = 0

    for Xn, Yn in dl_data:
        Xn = Xn.to(device)
        Yn = Yn.to(device)

        # Loss supervisionada em escala normalizada
        pred_data = net(Xn)
        loss_data = torch.mean((pred_data - Yn) ** 2)

        # Recupera X físico
        X_phys = unnormalize_x(Xn, x_sd, x_mu).detach()

        # x,y com gradiente para autograd
        x = X_phys[:, [feat_index["x"]]].clone().requires_grad_(True)
        y = X_phys[:, [feat_index["y"]]].clone().requires_grad_(True)

        # Reinjeta x,y no input físico
        X_phys_mod = X_phys.clone()
        X_phys_mod[:, [feat_index["x"]]] = x
        X_phys_mod[:, [feat_index["y"]]] = y

        # Renormaliza input
        Xn_mod = (X_phys_mod - x_mu) / x_sd

        # Predição normalizada e desnormalizada
        pred_phys_n = net(Xn_mod)
        pred_phys = pred_phys_n * y_sd + y_mu

        # Campos coarse físicos
        u_c = X_phys[:, [feat_index["Ux"]]]
        v_c = X_phys[:, [feat_index["Uy"]]]
        p_c = X_phys[:, [feat_index["p"]]]
        nut_c = X_phys[:, [feat_index["nut_log"]]]
        Re = X_phys[:, [feat_index["Re"]]]

        r_cont, r_mom_u, r_mom_v = pde_residuals_cont_mom(
            x=x,
            y=y,
            u_coarse=u_c,
            v_coarse=v_c,
            p_coarse=p_c,
            nut_coarse=nut_c,
            Re=Re,
            model_out=pred_phys
            , nut_transform=nut_transform
        )

        loss_cont = torch.mean(r_cont ** 2)

        huber = torch.nn.SmoothL1Loss(beta=1.0)
        loss_mom = huber(r_mom_u, torch.zeros_like(r_mom_u)) + huber(r_mom_v, torch.zeros_like(r_mom_v))

        loss = w_data * loss_data + w_cont * loss_cont + w_mom * loss_mom

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bs = Xn.size(0)
        total_loss += loss.item() * bs
        total_data += loss_data.item() * bs
        total_cont += loss_cont.item() * bs
        total_mom += loss_mom.item() * bs
        n_train += bs

    return (total_loss / n_train, 
            total_data / n_train, 
            total_cont / n_train,
            total_mom / n_train
    )


PINN_EPOCH_RUNNERS = {
    "continuity": train_pinn_epoch_continuity,
    "cont_mom": train_pinn_epoch_cont_mom,
}


def get_pinn_epoch_runner(physics_mode):
    if physics_mode not in PINN_EPOCH_RUNNERS:
        valid_modes = ", ".join(sorted(PINN_EPOCH_RUNNERS))
        raise ValueError(f"Unsupported PINN physics_mode '{physics_mode}'. Valid: {valid_modes}")
    return PINN_EPOCH_RUNNERS[physics_mode]
