from diffusers import DDPMPipeline  # 누락된 DDPMPipeline 임포트
from accelerate import Accelerator
from scripts.evaluate import evaluate
from tqdm.auto import tqdm

import os
import torch
import torch.nn.functional as F  # F를 torch.nn.functional로 임포트


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    
    project_dir = os.path.join(config.output_dir, "logs") 
    os.makedirs(project_dir, exist_ok=True)  # logs 프로젝트 디렉토리 생성
    
    #accelerator 초기화
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_dir="project_dir",
    )
    
    #분산학습환경에 맞게 acceleraotr을 준비 - 여러 GPU에서 병렬 처리 가능
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    #전체학습 loop에서 몇 번째 step인치 추적하기 위해 global_step을 설정
    #모니터링 할때 global_step이 기준이됨
    #한 epoches에서 뿐만 아니라 전체 스텝
    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        #step은 현재 배치가 몇번째인지를 나타내는 정수, 매 반복마다 1씩 증가
        #batch는 trian_dataloader에서 받아오는 데이터 묶음
        #batch를 통해 받아온 텐서에 무작위 노이지를 추가한뒤, 모델이 예측해야하는 noise_residual과 비교하여 손실(MSE)을 구함
        
        #########배치 처리###########
        #step : 학습루프에서 현재 배치의 index(몇번째 배치인지)
        #batch : 현재 배치의 데이터
        #train_dataloader에서 배치단위로 처리하는 loop


        for step, batch in enumerate(train_dataloader):
            
            #현재 배치에서 원본 이미지 데이터를 가져옴            
            clean_images = batch[0]
            
            # 랜덤 노이즈 생성
            # 원본 이미지와 동일한 크기의 랜덤 노이즈 텐서를 생성
            #.to 부분 : 생성된 노이즈 텐서를 이미지 텐서와 동일한 디바이스(GPU 또는 CPU)로 이동                        
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            
            #텐서에서 첫 번째 차원(배치 크기)을 가져옴
            #배치 크기 : 16
            bs = clean_images.shape[0]

            # 이미지에 대해 무작위로 타임스텝을 샘플링
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            #forward processing - 이미지에 노이즈를 추가가
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # 노이즈 잔여량 예측
                # 1.모델이 노이즈가 추가된 이미지(noisy_image)를 입력으로 받고 예측
                # 2.평균제곱오차를 총해 실제노이즈와 모델이 예측한 노이즈를 계산
                # 3.loss값을 기반으로 gradient 계산및 역전파 수행
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                #최적화 단계
                accelerator.clip_grad_norm_(model.parameters(), 1.0) #gradient bomb 방지
                optimizer.step() #parameter 업데이트
                lr_scheduler.step()
                optimizer.zero_grad()

            #TensorBoard에 손실과 학습률을 logging하고 진행상황 출력            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        #현재 실행하는 프로세스가 mainprocess인지
        #아래 작업을 단일 프로세스에서만 실행하도록 제한
        #분산환경에서의 작업 logic이해가 필요한 | 하나의 메인프로세스 + worker들
        #메인프로세스의 역할 : 전체 학습을 제어 (logging, 모델저장, 평가 등등)
        #worker들의 역할 : 실제 데이터 처리와 계산

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            #accelerator = Accelerator()
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.model_save_dir)