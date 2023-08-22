from mmcls.models import CLASSIFIERS, ImageClassifier


@CLASSIFIERS.register_module()
class ImageClassifierWithReturnFeature(ImageClassifier):
    def forward(self, x, *args, **kwargs):
        if 'return_patch_token' in kwargs:
            if kwargs['return_patch_token']:
                x = super().extract_feat(x)
                x = x[-1]
                patch_token, _ = x
                B, C, _, _ = patch_token.shape
                patch_token = patch_token.reshape(B, C, -1)
                avg_patch_token = patch_token.mean(dim=2)
                return avg_patch_token
        if 'return_feature' in kwargs:
            if kwargs['return_feature']:
                x = super().extract_feat(x)
                x = x[-1]
                _, cls_token = x
                cls_score = self.head.layers(cls_token)
                if isinstance(cls_score, list):
                    cls_score = sum(cls_score) / float(len(cls_score))
                return cls_score, cls_token
        if 'return_feature_list' in kwargs:
            if kwargs['return_feature_list']:
                self.backbone.out_indices = range(len(self.backbone.layers))
                outs = self.backbone(x)
                features_list = []
                for out in outs:
                    features_list.append(out[1])
                for out in outs:
                    B, C, _, _ = out[0].shape
                    patch_token = out[0].reshape(B, C, -1)
                    patch_token = patch_token.mean(dim=2)
                    features_list.append(patch_token)
                _, cls_token = outs[-1]
                cls_score = self.head.layers(cls_token)
                if isinstance(cls_score, list):
                    cls_score = sum(cls_score) / float(len(cls_score))
                return cls_score, features_list
        
        x = super().extract_feat(x)
        x = x[-1]
        _, cls_token = x
        cls_score = self.head.layers(cls_token)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        return cls_score

    def forward_threshold(self, x, threshold):
        x = super().extract_feat(x)
        _, cls_token = x[-1]
        cls_token = cls_token.clip(max=threshold)
        cls_score = self.head.layers(cls_token)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        return cls_score