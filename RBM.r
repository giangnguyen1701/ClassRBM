library(keras)
library(tidyverse)
library(dplyr)
library(caret)

## Used functions
target_transform <- function(y){ #transform categories to one-hot vector
  name = sort(unique(y))
  
  l = length(name)
  n = length(y)
  
  target = matrix(0, nrow = n, ncol = l)
  for (i in 1:l){
    target[y==name[i], i] = 1
  }
  colnames(target) <- name
  return(target)
}

fit_target <- function(y, n_y){ #transform from class (1,2,3,...) to one-hot vector
  n = length(y)
  target = matrix(0, n, n_y)
  for (i in 1:n_y){
    target[y==i, i] = 1
  }
  return(target)
}

sigm <- function(x){ #sigmoid function
  return(1/(1+exp(-x)))
}

softplus <- function(x){ #softplus function
  M = log(1+exp(x))
  is_inf = (M==Inf)
  M[M==Inf] = 0
  return(M + x*is_inf)
}

mini_batch <- function(n, batch.size){ #create mini batch
  id = vector(mode = 'list')
  l = ceiling(n/batch.size)
  if (l==1){
    id[[1]] = 1:n
  }
  else{
    x = sample(1:n)
    for (i in 1:(l-1)){
      id[[i]] = x[((i-1) * batch.size + 1) : (i * batch.size )]
    }
    id[[l]] = x[((l-1) * batch.size + 1) : n]
  }
  return(id)
}

generative_gradient <- function(x0, y0, W, U, b, c, d, alpha = 0){ #compute generative gradient
  n_b = dim(x0)[1]
  n_y = dim(y0)[2]
  n.hidden = length(c)
  h0 = t(sigm(W %*% t(x0) + U%*%t(y0)+ c))
  h0_s = (h0 > matrix(runif(n.hidden * n_b), n_b, n.hidden)) * 1
  
  p_y_h = t(apply(d + h0_s %*% U, 1, function(x) x-max(x)))
  p_y_h = t(apply(exp(p_y_h), 1, function(x) x/sum(x)))
  #p_y_h_c = t(apply(p_y_h, 1, cumsum))
  y1 = t(apply(p_y_h, 1, function(x) fit_target(sample(1:10,1,prob = x), n_y)))
  #y1 = rowSums(p_y_h_c < runif(n_b)) + 1
  #y1 = fit_target(y1, n_y)
  #y1 = p_y_h
  p_x_h = sigm(b + h0_s %*% W)
  #x1 = (p_x_h > matrix(runif(n_x * n), n_b, n_x)) * 1 
  x1 = p_x_h
  h1 = t(sigm(W%*%t(x1) + U%*%t(y1)+ c))
  
  grad_W = -(t(h0)%*%x0 - t(h1)%*%x1) / n_b
  grad_U = -(t(h0)%*%y0 - t(h1)%*%y1) / n_b
  grad_b = -colMeans(x0 - x1)
  grad_d = -colMeans(y0 - y1)
  grad_c = -colMeans(h0 - h1)
  
  return(list(grad_W = grad_W,
              grad_U = grad_U,
              grad_b = grad_b,
              grad_c = grad_c,
              grad_d = grad_d))
}

discriminative_gradient <- function(x0, y0, W, U, b, c, d, alpha = 0){ #compute discriminative gradient
  n_b = dim(x0)[1]
  n_y = dim(y0)[2]
  n.hidden = length(c)
  theta = W%*%t(x0)+ c
  
  h0 = t(sigm(theta + U%*%t(y0)))
  
  exp_free_energy = matrix(0,n_b,n_y)
  for (m in 1:n_y){
    y_s = matrix(0,n_b,n_y)
    y_s[,m]= 1
    exp_free_energy[,m] = d[m] + rowSums(softplus(t(theta + U%*%t(y_s))))
  }
  exp_free_energy = t(apply(exp_free_energy, 1, function(x) x-max(x)))
  p_y_x = t(apply(exp(exp_free_energy), 1, function(x) x/sum(x)))
  
  second_term_c = matrix(0,n.hidden,n_y)
  second_term_WU = matrix(0,n_b,n.hidden)
  
  for (m in 1:n_y){
    y_s = matrix(0,n_b,n_y)
    y_s[,m]= 1
    h_yx = t(sigm(theta + U%*%t(y_s)))
    sc = h_yx * p_y_x[,m]
    second_term_c[,m] = colSums(sc)
    second_term_WU = second_term_WU + sc
  }
  
  grad_c = -(colSums(h0) - rowSums(second_term_c)) / n_b
  grad_W = -t(h0 - second_term_WU)%*%x0 / n_b
  grad_U = -t(h0 - second_term_WU)%*%y0 / n_b
  grad_d = -colMeans(y0 - p_y_x)
  
  return(list(grad_W = grad_W,
              grad_U = grad_U,
              grad_b = rep(0, length(b)),
              grad_c = grad_c,
              grad_d = grad_d))
}

hybrid_gradient <- function(x0, y0, W, U, b, c, d, alpha = 0.01){ #compute hybrid gradient
  grad_discriminative = discriminative_gradient(x0, y0, W, U, b, c, d)
  grad_generative = generative_gradient(x0, y0, W, U, b, c, d)
  
  grad_W = grad_discriminative$grad_W + alpha * grad_generative$grad_W
  grad_U = grad_discriminative$grad_U + alpha * grad_generative$grad_U
  grad_b = grad_discriminative$grad_b + alpha * grad_generative$grad_b
  grad_d = grad_discriminative$grad_d + alpha * grad_generative$grad_d
  grad_c = grad_discriminative$grad_c + alpha * grad_generative$grad_c

  return(list(grad_W = grad_W,
              grad_U = grad_U,
              grad_b = grad_b,
              grad_c = grad_c,
              grad_d = grad_d))
}

predictRBM <- function(x, y, paras){ #classification prediction
  n.hidden = length(paras$c)
  h = rep(0, n.hidden)
  W = paras$W
  U = paras$U
  b = paras$b
  c = paras$c
  d = paras$d
  n_y = length(d)
  
  if (is.null(dim(x))){
    x = matrix(x, nrow = 1)
  }
  
  theta = W%*%t(x)+ c
  exp_free_energy = matrix(0,dim(x)[1],n_y)
  for (m in 1:n_y){
    y_s = matrix(0,dim(x)[1],n_y)
    y_s[,m]= 1
    exp_free_energy[,m] = d[m] + rowSums(softplus(t(theta + U%*%t(y_s))))
  }
  exp_free_energy = t(apply(exp_free_energy, 1, function(x) x-max(x)))
  p_y_x = t(apply(exp(exp_free_energy), 1, function(x) x/sum(x)))
  pred = apply(p_y_x, 1, function(x) which.max(x))
  class_pred = fit_target(pred, n_y)
  if (missing(y)){
    return(list(prob=p_y_x, class=class_pred))
  }
  else{
    if (is.null(dim(y))){
      y = matrix(y, nrow = 1)
    }
    train = apply(y, 1, function(x) which.max(x))
    pred = apply(p_y_x, 1, function(x) which.max(x))
    return(list(prob=p_y_x, class=class_pred, accuracy=sum(train==pred)))
  }
}

reconstructionRBM <- function(x, y , paras){ #reconstruction of observation with current parameters
  n.hidden = length(paras$c)
  h = rep(0, n.hidden)
  W = paras$W
  U = paras$U
  b = paras$b
  c = paras$c
  d = paras$d
  
  if (is.null(dim(x))){
    x = matrix(x, nrow = 1)
    y = matrix(y, nrow = 1)
  }
  
  n_b = dim(x)[1]
  n_y = dim(y)[2]
  h0 = t(sigm(W%*%t(x) + U%*%t(y)+ c))
  h0_s = (h0 > matrix(runif(n.hidden * n_b), n_b, n.hidden)) * 1
  p_x_h = sigm(b + h0_s %*% W)
  return(p_x_h)
}


# function for training RBM
RBM_training <- function(x, y, paras, 
                         n.hidden = 100, 
                         x_val, y_val, 
                         type = 'hybrid', #3 types: generative, discriminative and hybrid
                         alpha = 0.01, #for hybrid training
                         lr = 0.01,
                         n.epoch = 100,
                         batch.size = 100, 
                         opt = 'Adam', #3 types of optimizer: GD (gradient descent), Momentum and Adam
                         mu = 0.9, #for Momentum
                         beta1 = 0.9, #for Adam
                         beta2 = 0.999, #for Adam
                         eps = 1e-8, #for Adam
                         sample_rec = 27, #sample observation for inspecting reconstruction during Generative training
                         image.size = c(28,28) #size of image
                         ){
  n = dim(x)[1]
  n_x = dim(x)[2]
  n_y = dim(y)[2]
  
  if (missing(paras)){
    h = rep(0, n.hidden)
    W = matrix(runif(n.hidden * n_x), nrow = n.hidden, ncol = n_x) / 100 - 0.5/100
    U = matrix(runif(n.hidden * n_y), nrow = n.hidden, ncol = n_y) / 100 - 0.5/100
    b = runif(n_x) / 100 - 0.5/100
    d = runif(n_y) / 100 - 0.5/100
    c = runif(n.hidden) / 100 - 0.5/100
  }
  else{
    n.hidden = length(paras$c)
    h = rep(0, n.hidden)
    W = paras$W
    U = paras$U
    b = paras$b
    c = paras$c
    d = paras$d
    }
  gradient = get(paste(type, 'gradient', sep="_", collapse = NULL))
  # initial values for optimizer
  m_W <- m_U <- m_b <- m_c <- m_d <-0
  v_W <- v_U <- v_b <- v_c <- v_d <-0
  u_W <- u_U <- u_b <- u_c <- u_d <-0
  
  train_acc = vector(mode='numeric')
  val_acc = vector(mode='numeric')
  
  reconstruction_sample = NULL
  #Reconstruction sample for Generative traning
  if (type == 'generative'){
    current_rec = reconstructionRBM(x[sample_rec,], y[sample_rec,], 
                                    list(W = W, U = U, b = b, c = c, d = d))
    par(mfrow=c(1,2))
    image(matrix(x[sample_rec,], nrow = image.size[1]), ylim=c(1,0))
    image(matrix(current_rec, nrow = image.size[1]), ylim=c(1,0))
    reconstruction_sample = current_rec
  }
  
  for (i in 1:n.epoch){
    id <- mini_batch(n, batch.size)
    for (k in 1:length(id)){
      x0 = x[id[[k]],]
      y0 = y[id[[k]],]
      if (batch.size == 1){
        x0 = matrix(x0, nrow = 1)
        y0 = matrix(y0, nrow = 1)
      }
      # Update parameters
      grad = gradient(x0, y0, W, U, b, c, d, alpha)
      
      if (opt == 'Adam'){
        m_W = beta1*m_W + (1-beta1)*grad$grad_W
        mt_W = m_W / (1-beta1^i)
        v_W = beta2*v_W + (1-beta2)*(grad$grad_W^2)
        vt_W = v_W / (1-beta2^i)
        u_W = -lr * mt_W / (sqrt(vt_W) + eps)
        
        m_U = beta1*m_U + (1-beta1)*grad$grad_U
        mt_U = m_U / (1-beta1^i)
        v_U = beta2*v_U + (1-beta2)*(grad$grad_U^2)
        vt_U = v_U / (1-beta2^i)
        u_U = -lr * mt_U / (sqrt(vt_U) + eps)
        
        m_b = beta1*m_b + (1-beta1)*grad$grad_b
        mt_b = m_b / (1-beta1^i)
        v_b = beta2*v_b + (1-beta2)*(grad$grad_b^2)
        vt_b = v_b / (1-beta2^i)
        u_b = -lr * mt_b / (sqrt(vt_b) + eps)
        
        m_c = beta1*m_c + (1-beta1)*grad$grad_c
        mt_c = m_c / (1-beta1^i)
        v_c = beta2*v_c + (1-beta2)*(grad$grad_c^2)
        vt_c = v_c / (1-beta2^i)
        u_c = -lr * mt_c / (sqrt(vt_c) + eps)
        
        m_d = beta1*m_d + (1-beta1)*grad$grad_d
        mt_d = m_d / (1-beta1^i)
        v_d = beta2*v_d + (1-beta2)*(grad$grad_d^2)
        vt_d = v_d / (1-beta2^i)
        u_d = -lr * mt_d / (sqrt(vt_d) + eps)
        
      }
      
      else if (opt == 'GD'){
        u_W = - lr * grad$grad_W
        u_U = - lr * grad$grad_U
        u_b = - lr * grad$grad_b
        u_d = - lr * grad$grad_d
        u_c = - lr * grad$grad_c
      }
      
      
      else if (opt == 'Momentum'){
        u_W = mu * u_W - lr * grad$grad_W
        u_U = mu * u_U - lr * grad$grad_U
        u_b = mu * u_b - lr * grad$grad_b
        u_d = mu * u_d - lr * grad$grad_d
        u_c = mu * u_c - lr * grad$grad_c
      }
      
      W = W + u_W
      U = U + u_U
      b = b + u_b
      c = c + u_c
      d = d + u_d
      
    }
    print(paste('Epoch', i, sep = " "))
    #Training accuracy
    pred = predictRBM(x, y, list(W = W, U = U, b = b, c = c, d = d))
    train_acc = append(train_acc, pred$accuracy/n)
    print(paste('Training accuracy:', tail(train_acc,1)))
    
    #Validation acurracy
    if(!missing(x_val)){
      pred = predictRBM(x_val, y_val, list(W = W, U = U, b = b, c = c, d = d))
      val_acc = append(val_acc, pred$accuracy/dim(x_val)[1])
      print(paste('Validation accuracy:', tail(val_acc,1)))
    }
    
    #Reconstruction sample for Generative traning
    if (type == 'generative'){
      current_rec = reconstructionRBM(x[sample_rec,], y[sample_rec,], 
                                      list(W = W, U = U, b = b, c = c, d = d))
      par(mfrow=c(1,2))
      image(matrix(x[sample_rec,], nrow = 28), ylim=c(1,0))
      image(matrix(current_rec, nrow = 28), ylim=c(1,0))
      reconstruction_sample = rbind(reconstruction_sample, current_rec)
    }
  }
  paras = list(W = W, U = U, b = b, c = c, d = d)
  return(list(paras = paras,
              train_acc = train_acc,
              val_acc = val_acc,
              reconstruction_sample = reconstruction_sample))
}

# function for semi-supervised traning
semiRBM <- function(x, y, paras, 
                    x_unlabel, 
                    n.hidden = 100, 
                    x_val, y_val, 
                    beta = 0.1, 
                    type = 'discriminative', #3 types: generative, discriminative and hybrid
                    alpha = 0.01, #for hybrid training
                    lr = 0.01,
                    n.epoch = 100,
                    batch.size = 100, 
                    opt = 'Adam', #3 types of optimizer: GD (gradient descent), Momentum and Adam
                    mu = 0.9, #for Momentum
                    beta1 = 0.9, #for Adam
                    beta2 = 0.999, #for Adam
                    eps = 1e-8 #for Adam
                    
){
  n = dim(x)[1]
  n_x = dim(x)[2]
  n_y = dim(y)[2]
  
  if (missing(paras)){
    h = rep(0, n_hidden)
    W = matrix(runif(n_hidden * n_x), nrow = n_hidden, ncol = n_x) / 100 - 0.5/100
    U = matrix(runif(n_hidden * n_y), nrow = n_hidden, ncol = n_y) / 100 - 0.5/100
    b = runif(n_x) / 100 - 0.5/100
    d = runif(n_y) / 100 - 0.5/100
    c = runif(n_hidden) / 100 - 0.5/100
  }
  else{
    n.hidden = length(paras$c)
    h = rep(0, n_hidden)
    W = paras$W
    U = paras$U
    b = paras$b
    c = paras$c
    d = paras$d
  }
  n_unlabel = dim(x_unlabel)[1]
  
  gradient = get(paste(type, 'gradient', sep="_", collapse = NULL))
  # initial values for optimizer
  m_W <- m_U <- m_b <- m_c <- m_d <-0
  v_W <- v_U <- v_b <- v_c <- v_d <-0
  u_W <- u_U <- u_b <- u_c <- u_d <-0
  
  train_acc = vector(mode='numeric')
  val_acc = vector(mode='numeric')
  
  for (i in 1:n.epoch){
    id <- mini_batch(n_unlabel, batch.size)
    for (k in 1:length(id)){
      x0 = x_unlabel[id[[k]],]
      n_b = dim(x0)[1]
      # Update parameters
      #Gradient of supervised part
      grad_sup = gradient(x, y, W, U, b, c, d, alpha)
      
      # Gradient of unsupervised part
      
      p_y_x = predictRBM(x=x0, paras = list(W = W, U = U, b = b, c = c, d = d))$prob
      y0 = t(apply(p_y_x, 1, function(x) fit_target(sample(1:n_y,1,prob = x), n_y)))
      
      h0 = t(sigm(W%*%t(x0) + U%*%t(y0)+ c))
      h0_s = (h0 > matrix(runif(n_hidden * n_b), n_b, n_hidden)) * 1
      
      p_y_h = t(apply(d + h0_s %*% U, 1, function(x) x-max(x)))
      p_y_h = t(apply(exp(p_y_h), 1, function(x) x/sum(x)))
      p_y_h_c = t(apply(p_y_h, 1, cumsum))
      y1 = t(apply(p_y_h, 1, function(x) fit_target(sample(1:10,1,prob = x), n_y)))
      #y1 = rowSums(p_y_h_c < runif(n_b)) + 1
      #y1 = fit_target(y1, n_y)
      #y1 = p_y_h
      p_x_h = sigm(b + h0_s %*% W)
      #x1 = (p_x_h > matrix(runif(n_x * n), n_b, n_x)) * 1 
      x1 = p_x_h
      h1 = t(sigm(W%*%t(x1) + U%*%t(y1)+ c))
      
      theta = W%*%t(x0)+ c
      
      second_term_c = matrix(0,n_hidden,n_y)
      second_term_WU = matrix(0,n_b,n_hidden)
      
      for (m in 1:n_y){
        y_s = matrix(0,n_b,n_y)
        y_s[,m]= 1
        h_yx = t(sigm(theta + U%*%t(y_s)))
        sc = h_yx * p_y_x[,m]
        second_term_c[,m] = colSums(sc)
        second_term_WU = second_term_WU + sc
      }
      
      
      grad_unsup_W = (t(h1)%*%x1 - t(second_term_WU)%*%x0) / n_b
      grad_unsup_U = (t(h1)%*%y1 - t(second_term_WU)%*%y0) / n_b
      grad_unsup_b = colSums(x1 - x0) / n_b
      grad_unsup_d = colSums(y1 - p_y_x) / n_b
      grad_unsup_c = (colSums(h1) - rowSums(second_term_c)) / n_b
      
      grad = list(grad_W = grad_sup$grad_W  + beta * grad_unsup_W,
                  grad_U = grad_sup$grad_U  + beta * grad_unsup_U,
                  grad_b = grad_sup$grad_b  + beta * grad_unsup_b,
                  grad_c = grad_sup$grad_c  + beta * grad_unsup_c,
                  grad_d = grad_sup$grad_d  + beta * grad_unsup_d)
      
      if (opt == 'Adam'){
        m_W = beta1*m_W + (1-beta1)*grad$grad_W
        mt_W = m_W / (1-beta1^i)
        v_W = beta2*v_W + (1-beta2)*(grad$grad_W^2)
        vt_W = v_W / (1-beta2^i)
        u_W = -lr * mt_W / (sqrt(vt_W) + eps)
        
        m_U = beta1*m_U + (1-beta1)*grad$grad_U
        mt_U = m_U / (1-beta1^i)
        v_U = beta2*v_U + (1-beta2)*(grad$grad_U^2)
        vt_U = v_U / (1-beta2^i)
        u_U = -lr * mt_U / (sqrt(vt_U) + eps)
        
        m_b = beta1*m_b + (1-beta1)*grad$grad_b
        mt_b = m_b / (1-beta1^i)
        v_b = beta2*v_b + (1-beta2)*(grad$grad_b^2)
        vt_b = v_b / (1-beta2^i)
        u_b = -lr * mt_b / (sqrt(vt_b) + eps)
        
        m_c = beta1*m_c + (1-beta1)*grad$grad_c
        mt_c = m_c / (1-beta1^i)
        v_c = beta2*v_c + (1-beta2)*(grad$grad_c^2)
        vt_c = v_c / (1-beta2^i)
        u_c = -lr * mt_c / (sqrt(vt_c) + eps)
        
        m_d = beta1*m_d + (1-beta1)*grad$grad_d
        mt_d = m_d / (1-beta1^i)
        v_d = beta2*v_d + (1-beta2)*(grad$grad_d^2)
        vt_d = v_d / (1-beta2^i)
        u_d = -lr * mt_d / (sqrt(vt_d) + eps)
        
      }
      
      else if (opt == 'GD'){
        u_W = - lr * grad$grad_W
        u_U = - lr * grad$grad_U
        u_b = - lr * grad$grad_b
        u_d = - lr * grad$grad_d
        u_c = - lr * grad$grad_c
      }
      
      
      else if (opt == 'Momentum'){
        u_W = mu * u_W - lr * grad$grad_W
        u_U = mu * u_U - lr * grad$grad_U
        u_b = mu * u_b - lr * grad$grad_b
        u_d = mu * u_d - lr * grad$grad_d
        u_c = mu * u_c - lr * grad$grad_c
      }
      
      W = W + u_W
      U = U + u_U
      b = b + u_b
      c = c + u_c
      d = d + u_d
      
    }
    print(paste('Epoch', i, sep = " "))
    #Training accuracy
    pred = predictRBM(x, y, list(W = W, U = U, b = b, c = c, d = d))
    train_acc = append(train_acc, pred$accuracy)
    print(paste('Training accuracy:', tail(train_acc,1)/n))
    
    #Validation acurracy
    if(!missing(x_val)){
      pred = predictRBM(x_val, y_val, list(W = W, U = U, b = b, c = c, d = d))
      val_acc = append(val_acc, pred$accuracy)
      print(paste('Validation accuracy:', tail(val_acc,1)/dim(x_val)[1]))
    }
  }
  paras = list(W = W, U = U, b = b, c = c, d = d)
  return(list(paras = paras,
              train_acc = train_acc,
              val_acc = val_acc))
}

# Training with 10.000 observation:
mnist <- dataset_mnist()
img_rows <- 28
img_cols <- 28

x <- (mnist$train$x/255) %>% array_reshape(c(nrow(mnist$train$x), img_rows * img_cols)) 
x_test <- (mnist$test$x/255) %>% array_reshape(c(nrow(mnist$test$x), img_rows * img_cols)) 

#y_train <- target_transform(mnist$train$y)
y_test <- target_transform(mnist$test$y)

y <- mnist$train$y

cvIndex <- createFolds(y, 6, returnTrain = F)

x_val <- x[cvIndex[[2]],]
y_val <- y[cvIndex[[2]]]
y_val = target_transform(y_val)

x_train <- x[-cvIndex[[2]],]
y_train <- y[-cvIndex[[2]]]
y_train = target_transform(y_train)

rbm_gen = RBM_training(x = x_train, y = y_train,
            n.hidden = 100, 
            x_val = x_val, y_val = y_val, 
            type = 'generative',
            alpha = 0.01, 
            lr = 0.01,
            n.epoch = 100,
            batch.size = 1, 
            opt = 'Adam', 
            mu = 0.5, #for Momentum
            beta1 = 0.9, #for Adam
            beta2 = 0.999, #for Adam
            eps = 1e-8, #for Adam
            sample_rec = 8#sample observation for inspecting reconstruction during Generative training
)



rbm_dis = RBM_training(x = x_train, y = y_train,
                       n.hidden = 100, 
                       x_val = x_val, y_val = y_val, 
                       type = 'discriminative',
                       alpha = 0.01, 
                       lr = 0.01,
                       n.epoch = 100,
                       batch.size = 100, 
                       opt = 'Adam', 
                       mu = 0.5, #for Momentum
                       beta1 = 0.9, #for Adam
                       beta2 = 0.999, #for Adam
                       eps = 1e-8, #for Adam
                       sample_rec = 27#sample observation for inspecting reconstruction during Generative training
)

rbm_hyb = RBM_training(x = x_train, y = y_train, 
                       n.hidden = 100, 
                       x_val = x_val, y_val = y_val, 
                       type = 'hybrid',
                       alpha = 0.01, 
                       lr = 0.01,
                       n.epoch = 100,
                       batch.size = 100, 
                       opt = 'Adam', 
                       mu = 0.5, #for Momentum
                       beta1 = 0.9, #for Adam
                       beta2 = 0.999, #for Adam
                       eps = 1e-8, #for Adam
                       sample_rec = 27#sample observation for inspecting reconstruction during Generative training
)

# plot for generative traning
windows(10,4)
par(mfrow=c(2,5), cex=0.7, mai=c(0.1,0.1,0.2,0.1))
for (ii in c(0:9)){
  image(matrix(rbm_gen$reconstruction_sample[ii+1,], nrow = 28), axes = F, ylim=c(1,0), main = paste("Epoch ", ii))
}

windows(10,4)
par(mfrow=c(2,5), cex=0.7, mai=c(0.2,0.2,0.3,0.2))
for (ii in seq(10,100,10)){
  image(matrix(rbm_gen$reconstruction_sample[ii+1,], nrow = 28), axes = F, ylim=c(1,0),main = paste("Epoch ", ii))
}

plot(c(0.1,rbm_gen$train_acc), col = 'red', type = 'l', xlab = 'Epoch', ylab = 'Accuracy', ylim = c(0,1))
lines(c(0.1,rbm_gen$val_acc), col = 'blue')
legend(80,0.5,legend=c("train","validation"), col=c("red","blue"), lty=c(1,1), cex=0.8)

# plot for discriminative and hybrid training
windows(10,3)
par(mfrow=c(1,2), cex=0.7)
plot(c(rbm_dis$train_acc), col = 'red', type = 'l', xlab = 'Epoch', ylab = 'Accuracy', ylim = c(0,1), main = 'Discriminative')
lines(c(rbm_dis$val_acc), col = 'blue')
lines(rep(0.945, 100), col = 'black', lty=2)
legend(80,0.98,legend=c("train","validation"), col=c("red","blue"), lty=c(1,1), cex=0.8)
plot(rbm_hyb$train_acc, col = 'red', type = 'l', xlab = 'Epoch', ylab = 'Accuracy', ylim = c(0,1), main = 'Hybrid')
lines(rbm_hyb$val_acc, col = 'blue')
lines(rep(0.945, 200), col = 'black', lty=2)
legend(160,0.98,legend=c("train","validation"), col=c("red","blue"), lty=c(1,1), cex=0.8)



#semi-supervised
n_hidden = 100
n = dim(x_train)[1]
n_x = dim(x_train)[2]
n_y = dim(y_train)[2]

h = rep(0, n_hidden)
W = matrix(runif(n_hidden * n_x), nrow = n_hidden, ncol = n_x) / 100 - 0.5/100
U = matrix(runif(n_hidden * n_y), nrow = n_hidden, ncol = n_y) / 100 - 0.5/100
b = runif(n_x) / 100 - 0.5/100
d = runif(n_y) / 100 - 0.5/100
c = runif(n_hidden) / 100 - 0.5/100
paras = list(W = W, U = U, b = b, c = c, d = d)

rbm_semi = semiRBM(x = x_train[1:1000,], y = y_train[1:1000,], paras = paras, 
                   x_unlabel = x_train[1001:50000,], 
                   n.hidden = 100, 
                   x_val, y_val, 
                   beta = 0.1,
                   type = 'hybrid', #3 types: generative, discriminative and hybrid
                   alpha = 0.1, #for hybrid training
                   lr = 0.005,
                   n.epoch = 20,
                   batch.size = 1000, 
                   opt = 'Adam', #3 types of optimizer: GD (gradient descent), Momentum and Adam
                   mu = 0.5, #for Momentum
                   beta1 = 0.9, #for Adam
                   beta2 = 0.999, #for Adam
                   eps = 1e-12 #for Adam
                   
)