목차
자기소개
Object Detection 소개
	Object Detection의 정의
	Deep learning 이전의 방식
	최근의 방식
		공통구조
		single stage
		two stage
	COCO Dataset 소개
	MMDetection Framework 소개
	Object Detection의 key factors
		Scale invariance
		Translation invariance
	Object Detection에서 Deep learning의 중요성
		무엇을 학습할 것인가?
		Anchor의 탄생
		Anchor의 작동방식
environment setup
	mmcv
	repo clone
	dataset download
Build modules
	build data loader
	build model
	load pretrained weights
forward train
	load data
	feature extract
	bbox prediction
	get loss
		get anchor_generator(anchor 그리는 규칙을 함수화)
			get base_anchors (기본 anchor 그리기)
			visualize base_anchors
		get anchors
			grid_anchor(기본 anchor를 stride만큼씩 띄워서 뿌리기)
			valid_flag(feature map에서 벗어나는 위치의 anchor를 걸러내기)
		make target(anchor target)
			except bboxes out of image(inside_flag 계산)
			assign
				bbox_overlaps(bbox 그룹간의 iou구하기)
				assign_wrt_overlaps(iou를 바탕으로 positive / negative sample 고르기)
			sampling(학습에 사용할 bbox를 sampling하기 - RetinaNet은 제외)
			make target with delta
				target initialization(flat anchors와 같은 크기)
				bbox2delta(anchor와 gt의 차이)
				get (bbox, labels) targets(delta와 label을 저장한다)
				label의 channel 펼치기
				unmap targets(inside_flag 적용하기 전 상태로 되돌린다)
			images_to_levels (이미지 별로 펼친 target을 level별로 쌓아 올린다)
		get loss
			cls_loss : focal loss
			bbox_loss : smooth_l1 loss
forward test
	get bbox
		delta2bbox
		multi_class_nms
	get result
		arrange results
	show result
remark
Q & A
