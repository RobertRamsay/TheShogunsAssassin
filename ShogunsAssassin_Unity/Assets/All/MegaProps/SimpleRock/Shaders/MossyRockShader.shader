// Upgrade NOTE: replaced '_Object2World' with 'unity_ObjectToWorld'
// Upgrade NOTE: replaced '_World2Object' with 'unity_WorldToObject'

// Shader created with Shader Forge v1.30 
// Shader Forge (c) Neat Corporation / Joachim Holmer - http://www.acegikmo.com/shaderforge/
// Note: Manually altering this data may prevent you from opening it in Shader Forge
/*SF_DATA;ver:1.30;sub:START;pass:START;ps:flbk:,iptp:0,cusa:False,bamd:0,lico:1,lgpr:1,limd:3,spmd:0,trmd:0,grmd:0,uamb:True,mssp:True,bkdf:True,hqlp:True,rprd:True,enco:False,rmgx:True,rpth:0,vtps:0,hqsc:True,nrmq:1,nrsp:0,vomd:0,spxs:False,tesm:1,olmd:1,culm:0,bsrc:0,bdst:1,dpts:2,wrdp:True,dith:0,rfrpo:True,rfrpn:Refraction,coma:15,ufog:True,aust:True,igpj:False,qofs:0,qpre:2,rntp:3,fgom:False,fgoc:False,fgod:False,fgor:False,fgmd:0,fgcr:0.5,fgcg:0.5,fgcb:0.5,fgca:1,fgde:0.01,fgrn:0,fgrf:300,stcl:False,stva:128,stmr:255,stmw:255,stcp:6,stps:0,stfa:0,stfz:0,ofsf:0,ofsu:0,f2p0:False,fnsp:False,fnfb:False;n:type:ShaderForge.SFN_Final,id:4218,x:36761,y:33272,varname:node_4218,prsc:2|diff-4309-OUT,spec-9574-OUT,gloss-5705-OUT,normal-9635-OUT,clip-6011-A;n:type:ShaderForge.SFN_Tex2d,id:770,x:34651,y:32977,ptovrint:False,ptlb:BaseTexture,ptin:_BaseTexture,varname:node_770,prsc:2,glob:False,taghide:False,taghdr:False,tagprd:False,tagnsco:False,tagnrm:False,ntxv:0,isnm:False;n:type:ShaderForge.SFN_Color,id:3571,x:34915,y:33017,ptovrint:False,ptlb:SpecColGloss(A),ptin:_SpecColGlossA,varname:node_3571,prsc:2,glob:False,taghide:False,taghdr:False,tagprd:False,tagnsco:False,tagnrm:False,c1:0.5,c2:0.5,c3:0.5,c4:1;n:type:ShaderForge.SFN_Tex2d,id:8495,x:34363,y:33202,ptovrint:False,ptlb:NormalMap,ptin:_NormalMap,varname:node_8495,prsc:2,glob:False,taghide:False,taghdr:False,tagprd:False,tagnsco:False,tagnrm:False,ntxv:3,isnm:True;n:type:ShaderForge.SFN_Color,id:6011,x:34779,y:32824,ptovrint:False,ptlb:BaseColourOpac(A),ptin:_BaseColourOpacA,varname:node_6011,prsc:2,glob:False,taghide:False,taghdr:False,tagprd:False,tagnsco:False,tagnrm:False,c1:0.5,c2:0.5,c3:0.5,c4:1;n:type:ShaderForge.SFN_Multiply,id:3062,x:35026,y:32824,varname:node_3062,prsc:2|A-6011-RGB,B-770-RGB;n:type:ShaderForge.SFN_Tex2d,id:9722,x:34915,y:33249,ptovrint:False,ptlb:Texture_SpecColorGloss(A),ptin:_Texture_SpecColorGlossA,varname:node_9722,prsc:2,glob:False,taghide:False,taghdr:False,tagprd:False,tagnsco:False,tagnrm:False,ntxv:0,isnm:False;n:type:ShaderForge.SFN_Multiply,id:6116,x:35164,y:32969,varname:node_6116,prsc:2|A-3571-RGB,B-9722-RGB;n:type:ShaderForge.SFN_Multiply,id:2851,x:35165,y:33165,varname:node_2851,prsc:2|A-3571-A,B-9722-A;n:type:ShaderForge.SFN_NormalVector,id:4007,x:34184,y:32607,prsc:2,pt:False;n:type:ShaderForge.SFN_ComponentMask,id:4118,x:34500,y:32613,varname:node_4118,prsc:2,cc1:1,cc2:-1,cc3:-1,cc4:-1|IN-4007-OUT;n:type:ShaderForge.SFN_Multiply,id:1966,x:35627,y:32850,varname:node_1966,prsc:2|A-3239-OUT,B-3062-OUT;n:type:ShaderForge.SFN_Add,id:1588,x:35211,y:32679,varname:node_1588,prsc:2|A-4314-OUT,B-4118-OUT;n:type:ShaderForge.SFN_Slider,id:1847,x:34470,y:32522,ptovrint:False,ptlb:AdditonalGradient,ptin:_AdditonalGradient,varname:node_1847,prsc:2,glob:False,taghide:False,taghdr:False,tagprd:False,tagnsco:False,tagnrm:False,min:-1,cur:2,max:3;n:type:ShaderForge.SFN_Multiply,id:2187,x:35026,y:32597,varname:node_2187,prsc:2|A-1847-OUT,B-618-OUT;n:type:ShaderForge.SFN_Multiply,id:9635,x:34678,y:33382,varname:node_9635,prsc:2|A-8495-RGB,B-2369-OUT;n:type:ShaderForge.SFN_Vector3,id:2369,x:34511,y:33553,varname:node_2369,prsc:2,v1:3,v2:3,v3:1;n:type:ShaderForge.SFN_ConstantClamp,id:3239,x:35380,y:32735,varname:node_3239,prsc:2,min:0.5,max:3|IN-1588-OUT;n:type:ShaderForge.SFN_Multiply,id:9574,x:35682,y:33008,varname:node_9574,prsc:2|A-3239-OUT,B-6116-OUT;n:type:ShaderForge.SFN_Multiply,id:5705,x:35682,y:33178,varname:node_5705,prsc:2|A-3239-OUT,B-2851-OUT;n:type:ShaderForge.SFN_Multiply,id:4467,x:35977,y:33238,varname:node_4467,prsc:2|A-1966-OUT,B-1850-OUT;n:type:ShaderForge.SFN_Multiply,id:2521,x:35316,y:33359,varname:node_2521,prsc:2|A-1847-OUT,B-8495-B;n:type:ShaderForge.SFN_Multiply,id:1850,x:35562,y:33473,varname:node_1850,prsc:2|A-2521-OUT,B-3856-OUT;n:type:ShaderForge.SFN_ConstantClamp,id:3856,x:35316,y:33540,varname:node_3856,prsc:2,min:0.5,max:1|IN-8495-G;n:type:ShaderForge.SFN_Tex2dAsset,id:5990,x:33953,y:33963,ptovrint:False,ptlb:OnTopTexture(maskA),ptin:_OnTopTexturemaskA,varname:node_5990,glob:False,taghide:False,taghdr:False,tagprd:False,tagnsco:True,tagnrm:False,ntxv:0,isnm:False;n:type:ShaderForge.SFN_Tex2d,id:4753,x:35457,y:33850,varname:node_4753,prsc:2,ntxv:0,isnm:False|UVIN-3825-OUT,TEX-5990-TEX;n:type:ShaderForge.SFN_Tex2d,id:2083,x:35457,y:33989,varname:node_2083,prsc:2,ntxv:0,isnm:False|UVIN-9514-OUT,TEX-5990-TEX;n:type:ShaderForge.SFN_Slider,id:3674,x:34368,y:34235,ptovrint:False,ptlb:MaskTiling,ptin:_MaskTiling,varname:node_3674,prsc:2,glob:False,taghide:False,taghdr:False,tagprd:False,tagnsco:False,tagnrm:False,min:0.01,cur:1,max:8;n:type:ShaderForge.SFN_Append,id:1272,x:35051,y:34148,varname:node_1272,prsc:2|A-3674-OUT,B-3674-OUT;n:type:ShaderForge.SFN_Slider,id:8654,x:34319,y:33732,ptovrint:False,ptlb:TopTextureTiling,ptin:_TopTextureTiling,varname:node_8654,prsc:2,glob:False,taghide:False,taghdr:False,tagprd:False,tagnsco:False,tagnrm:False,min:0.1,cur:1,max:32;n:type:ShaderForge.SFN_Append,id:2379,x:34899,y:33700,varname:node_2379,prsc:2|A-8654-OUT,B-8654-OUT;n:type:ShaderForge.SFN_Lerp,id:4309,x:36295,y:33808,varname:node_4309,prsc:2|A-4467-OUT,B-2505-OUT,T-2351-OUT;n:type:ShaderForge.SFN_Multiply,id:3825,x:35224,y:33747,varname:node_3825,prsc:2|A-2379-OUT,B-2005-UVOUT;n:type:ShaderForge.SFN_Multiply,id:9514,x:35190,y:34046,varname:node_9514,prsc:2|A-3616-UVOUT,B-1272-OUT;n:type:ShaderForge.SFN_TexCoord,id:3616,x:34923,y:34000,varname:node_3616,prsc:2,uv:0;n:type:ShaderForge.SFN_TexCoord,id:2005,x:34742,y:33829,varname:node_2005,prsc:2,uv:0;n:type:ShaderForge.SFN_Slider,id:6414,x:34384,y:34462,ptovrint:False,ptlb:TopTextureAmount,ptin:_TopTextureAmount,varname:node_6414,prsc:2,glob:False,taghide:False,taghdr:False,tagprd:False,tagnsco:False,tagnrm:False,min:0,cur:2,max:2;n:type:ShaderForge.SFN_Multiply,id:8447,x:35457,y:34153,varname:node_8447,prsc:2|A-4118-OUT,B-6414-OUT;n:type:ShaderForge.SFN_Multiply,id:2862,x:35773,y:34106,varname:node_2862,prsc:2|A-2083-A,B-8447-OUT;n:type:ShaderForge.SFN_Clamp01,id:2351,x:36047,y:34070,varname:node_2351,prsc:2|IN-2862-OUT;n:type:ShaderForge.SFN_Color,id:1776,x:35457,y:33685,ptovrint:False,ptlb:TopTextureColour,ptin:_TopTextureColour,varname:node_1776,prsc:2,glob:False,taghide:False,taghdr:False,tagprd:False,tagnsco:False,tagnrm:False,c1:0.5,c2:0.5,c3:0.5,c4:1;n:type:ShaderForge.SFN_Multiply,id:2505,x:35671,y:33721,varname:node_2505,prsc:2|A-1776-RGB,B-4753-RGB;n:type:ShaderForge.SFN_Add,id:4314,x:35272,y:32477,varname:node_4314,prsc:2|A-1847-OUT,B-2187-OUT;n:type:ShaderForge.SFN_ObjectScale,id:8966,x:34271,y:32828,varname:node_8966,prsc:2,rcp:False;n:type:ShaderForge.SFN_Divide,id:618,x:34733,y:32670,varname:node_618,prsc:2|A-4118-OUT,B-8966-Y;proporder:6011-770-3571-9722-8495-1847-5990-1776-8654-3674-6414;pass:END;sub:END;*/

Shader "Custom/MossyRockShader" {
    Properties {
        _BaseColourOpacA ("BaseColourOpac(A)", Color) = (0.5,0.5,0.5,1)
        _BaseTexture ("BaseTexture", 2D) = "white" {}
        _SpecColGlossA ("SpecColGloss(A)", Color) = (0.5,0.5,0.5,1)
        _Texture_SpecColorGlossA ("Texture_SpecColorGloss(A)", 2D) = "white" {}
        _NormalMap ("NormalMap", 2D) = "bump" {}
        _AdditonalGradient ("AdditonalGradient", Range(-1, 3)) = 2
        [NoScaleOffset]_OnTopTexturemaskA ("OnTopTexture(maskA)", 2D) = "white" {}
        _TopTextureColour ("TopTextureColour", Color) = (0.5,0.5,0.5,1)
        _TopTextureTiling ("TopTextureTiling", Range(0.1, 32)) = 1
        _MaskTiling ("MaskTiling", Range(0.01, 8)) = 1
        _TopTextureAmount ("TopTextureAmount", Range(0, 2)) = 2
        [HideInInspector]_Cutoff ("Alpha cutoff", Range(0,1)) = 0.5
    }
    SubShader {
        Tags {
            "Queue"="AlphaTest"
            "RenderType"="TransparentCutout"
        }
        LOD 200
        Pass {
            Name "FORWARD"
            Tags {
                "LightMode"="ForwardBase"
            }
            
            
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #define UNITY_PASS_FORWARDBASE
            #define SHOULD_SAMPLE_SH ( defined (LIGHTMAP_OFF) && defined(DYNAMICLIGHTMAP_OFF) )
            #define _GLOSSYENV 1
            #include "UnityCG.cginc"
            #include "AutoLight.cginc"
            #include "Lighting.cginc"
            #include "UnityPBSLighting.cginc"
            #include "UnityStandardBRDF.cginc"
            #pragma multi_compile_fwdbase_fullshadows
            #pragma multi_compile LIGHTMAP_OFF LIGHTMAP_ON
            #pragma multi_compile DIRLIGHTMAP_OFF DIRLIGHTMAP_COMBINED DIRLIGHTMAP_SEPARATE
            #pragma multi_compile DYNAMICLIGHTMAP_OFF DYNAMICLIGHTMAP_ON
            #pragma multi_compile_fog
            #pragma exclude_renderers metal xbox360 xboxone ps3 ps4 psp2 
            #pragma target 3.0
            uniform sampler2D _BaseTexture; uniform float4 _BaseTexture_ST;
            uniform float4 _SpecColGlossA;
            uniform sampler2D _NormalMap; uniform float4 _NormalMap_ST;
            uniform float4 _BaseColourOpacA;
            uniform sampler2D _Texture_SpecColorGlossA; uniform float4 _Texture_SpecColorGlossA_ST;
            uniform float _AdditonalGradient;
            uniform sampler2D _OnTopTexturemaskA;
            uniform float _MaskTiling;
            uniform float _TopTextureTiling;
            uniform float _TopTextureAmount;
            uniform float4 _TopTextureColour;
            struct VertexInput {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float4 tangent : TANGENT;
                float2 texcoord0 : TEXCOORD0;
                float2 texcoord1 : TEXCOORD1;
                float2 texcoord2 : TEXCOORD2;
            };
            struct VertexOutput {
                float4 pos : SV_POSITION;
                float2 uv0 : TEXCOORD0;
                float2 uv1 : TEXCOORD1;
                float2 uv2 : TEXCOORD2;
                float4 posWorld : TEXCOORD3;
                float3 normalDir : TEXCOORD4;
                float3 tangentDir : TEXCOORD5;
                float3 bitangentDir : TEXCOORD6;
                LIGHTING_COORDS(7,8)
                UNITY_FOG_COORDS(9)
                #if defined(LIGHTMAP_ON) || defined(UNITY_SHOULD_SAMPLE_SH)
                    float4 ambientOrLightmapUV : TEXCOORD10;
                #endif
            };
            VertexOutput vert (VertexInput v) {
                VertexOutput o = (VertexOutput)0;
                o.uv0 = v.texcoord0;
                o.uv1 = v.texcoord1;
                o.uv2 = v.texcoord2;
                #ifdef LIGHTMAP_ON
                    o.ambientOrLightmapUV.xy = v.texcoord1.xy * unity_LightmapST.xy + unity_LightmapST.zw;
                    o.ambientOrLightmapUV.zw = 0;
                #endif
                #ifdef DYNAMICLIGHTMAP_ON
                    o.ambientOrLightmapUV.zw = v.texcoord2.xy * unity_DynamicLightmapST.xy + unity_DynamicLightmapST.zw;
                #endif
                o.normalDir = UnityObjectToWorldNormal(v.normal);
                o.tangentDir = normalize( mul( unity_ObjectToWorld, float4( v.tangent.xyz, 0.0 ) ).xyz );
                o.bitangentDir = normalize(cross(o.normalDir, o.tangentDir) * v.tangent.w);
                float3 recipObjScale = float3( length(unity_WorldToObject[0].xyz), length(unity_WorldToObject[1].xyz), length(unity_WorldToObject[2].xyz) );
                float3 objScale = 1.0/recipObjScale;
                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
                float3 lightColor = _LightColor0.rgb;
                o.pos = mul(UNITY_MATRIX_MVP, v.vertex );
                UNITY_TRANSFER_FOG(o,o.pos);
                TRANSFER_VERTEX_TO_FRAGMENT(o)
                return o;
            }
            float4 frag(VertexOutput i) : COLOR {
                float3 recipObjScale = float3( length(unity_WorldToObject[0].xyz), length(unity_WorldToObject[1].xyz), length(unity_WorldToObject[2].xyz) );
                float3 objScale = 1.0/recipObjScale;
                i.normalDir = normalize(i.normalDir);
                float3x3 tangentTransform = float3x3( i.tangentDir, i.bitangentDir, i.normalDir);
                float3 viewDirection = normalize(_WorldSpaceCameraPos.xyz - i.posWorld.xyz);
                float3 _NormalMap_var = UnpackNormal(tex2D(_NormalMap,TRANSFORM_TEX(i.uv0, _NormalMap)));
                float3 normalLocal = (_NormalMap_var.rgb*float3(3,3,1));
                float3 normalDirection = normalize(mul( normalLocal, tangentTransform )); // Perturbed normals
                float3 viewReflectDirection = reflect( -viewDirection, normalDirection );
                clip(_BaseColourOpacA.a - 0.5);
                float3 lightDirection = normalize(_WorldSpaceLightPos0.xyz);
                float3 lightColor = _LightColor0.rgb;
                float3 halfDirection = normalize(viewDirection+lightDirection);
////// Lighting:
                float attenuation = LIGHT_ATTENUATION(i);
                float3 attenColor = attenuation * _LightColor0.xyz;
                float Pi = 3.141592654;
                float InvPi = 0.31830988618;
///////// Gloss:
                float node_4118 = i.normalDir.g;
                float node_3239 = clamp(((_AdditonalGradient+(_AdditonalGradient*(node_4118/objScale.g)))+node_4118),0.5,3);
                float4 _Texture_SpecColorGlossA_var = tex2D(_Texture_SpecColorGlossA,TRANSFORM_TEX(i.uv0, _Texture_SpecColorGlossA));
                float gloss = (node_3239*(_SpecColGlossA.a*_Texture_SpecColorGlossA_var.a));
                float specPow = exp2( gloss * 10.0+1.0);
/////// GI Data:
                UnityLight light;
                #ifdef LIGHTMAP_OFF
                    light.color = lightColor;
                    light.dir = lightDirection;
                    light.ndotl = LambertTerm (normalDirection, light.dir);
                #else
                    light.color = half3(0.f, 0.f, 0.f);
                    light.ndotl = 0.0f;
                    light.dir = half3(0.f, 0.f, 0.f);
                #endif
                UnityGIInput d;
                d.light = light;
                d.worldPos = i.posWorld.xyz;
                d.worldViewDir = viewDirection;
                d.atten = attenuation;
                #if defined(LIGHTMAP_ON) || defined(DYNAMICLIGHTMAP_ON)
                    d.ambient = 0;
                    d.lightmapUV = i.ambientOrLightmapUV;
                #else
                    d.ambient = i.ambientOrLightmapUV;
                #endif
                d.boxMax[0] = unity_SpecCube0_BoxMax;
                d.boxMin[0] = unity_SpecCube0_BoxMin;
                d.probePosition[0] = unity_SpecCube0_ProbePosition;
                d.probeHDR[0] = unity_SpecCube0_HDR;
                d.boxMax[1] = unity_SpecCube1_BoxMax;
                d.boxMin[1] = unity_SpecCube1_BoxMin;
                d.probePosition[1] = unity_SpecCube1_ProbePosition;
                d.probeHDR[1] = unity_SpecCube1_HDR;
                Unity_GlossyEnvironmentData ugls_en_data;
                ugls_en_data.roughness = 1.0 - gloss;
                ugls_en_data.reflUVW = viewReflectDirection;
                UnityGI gi = UnityGlobalIllumination(d, 1, normalDirection, ugls_en_data );
                lightDirection = gi.light.dir;
                lightColor = gi.light.color;
////// Specular:
                float NdotL = max(0, dot( normalDirection, lightDirection ));
                float LdotH = max(0.0,dot(lightDirection, halfDirection));
                float3 specularColor = (node_3239*(_SpecColGlossA.rgb*_Texture_SpecColorGlossA_var.rgb));
                float specularMonochrome;
                float4 _BaseTexture_var = tex2D(_BaseTexture,TRANSFORM_TEX(i.uv0, _BaseTexture));
                float2 node_3825 = (float2(_TopTextureTiling,_TopTextureTiling)*i.uv0);
                float4 node_4753 = tex2D(_OnTopTexturemaskA,node_3825);
                float2 node_9514 = (i.uv0*float2(_MaskTiling,_MaskTiling));
                float4 node_2083 = tex2D(_OnTopTexturemaskA,node_9514);
                float3 diffuseColor = lerp(((node_3239*(_BaseColourOpacA.rgb*_BaseTexture_var.rgb))*((_AdditonalGradient*_NormalMap_var.b)*clamp(_NormalMap_var.g,0.5,1))),(_TopTextureColour.rgb*node_4753.rgb),saturate((node_2083.a*(node_4118*_TopTextureAmount)))); // Need this for specular when using metallic
                diffuseColor = EnergyConservationBetweenDiffuseAndSpecular(diffuseColor, specularColor, specularMonochrome);
                specularMonochrome = 1.0-specularMonochrome;
                float NdotV = max(0.0,dot( normalDirection, viewDirection ));
                float NdotH = max(0.0,dot( normalDirection, halfDirection ));
                float VdotH = max(0.0,dot( viewDirection, halfDirection ));
                float visTerm = SmithJointGGXVisibilityTerm( NdotL, NdotV, 1.0-gloss );
                float normTerm = max(0.0, GGXTerm(NdotH, 1.0-gloss));
                float specularPBL = (NdotL*visTerm*normTerm) * (UNITY_PI / 4);
                if (IsGammaSpace())
                    specularPBL = sqrt(max(1e-4h, specularPBL));
                specularPBL = max(0, specularPBL * NdotL);
                float3 directSpecular = (floor(attenuation) * _LightColor0.xyz)*specularPBL*FresnelTerm(specularColor, LdotH);
                half grazingTerm = saturate( gloss + specularMonochrome );
                float3 indirectSpecular = (gi.indirect.specular);
                indirectSpecular *= FresnelLerp (specularColor, grazingTerm, NdotV);
                float3 specular = (directSpecular + indirectSpecular);
/////// Diffuse:
                NdotL = max(0.0,dot( normalDirection, lightDirection ));
                half fd90 = 0.5 + 2 * LdotH * LdotH * (1-gloss);
                float nlPow5 = Pow5(1-NdotL);
                float nvPow5 = Pow5(1-NdotV);
                float3 directDiffuse = ((1 +(fd90 - 1)*nlPow5) * (1 + (fd90 - 1)*nvPow5) * NdotL) * attenColor;
                float3 indirectDiffuse = float3(0,0,0);
                indirectDiffuse += gi.indirect.diffuse;
                diffuseColor *= 1-specularMonochrome;
                float3 diffuse = (directDiffuse + indirectDiffuse) * diffuseColor;
/// Final Color:
                float3 finalColor = diffuse + specular;
                fixed4 finalRGBA = fixed4(finalColor,1);
                UNITY_APPLY_FOG(i.fogCoord, finalRGBA);
                return finalRGBA;
            }
            ENDCG
        }
        Pass {
            Name "FORWARD_DELTA"
            Tags {
                "LightMode"="ForwardAdd"
            }
            Blend One One
            
            
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #define UNITY_PASS_FORWARDADD
            #define SHOULD_SAMPLE_SH ( defined (LIGHTMAP_OFF) && defined(DYNAMICLIGHTMAP_OFF) )
            #define _GLOSSYENV 1
            #include "UnityCG.cginc"
            #include "AutoLight.cginc"
            #include "Lighting.cginc"
            #include "UnityPBSLighting.cginc"
            #include "UnityStandardBRDF.cginc"
            #pragma multi_compile_fwdadd_fullshadows
            #pragma multi_compile LIGHTMAP_OFF LIGHTMAP_ON
            #pragma multi_compile DIRLIGHTMAP_OFF DIRLIGHTMAP_COMBINED DIRLIGHTMAP_SEPARATE
            #pragma multi_compile DYNAMICLIGHTMAP_OFF DYNAMICLIGHTMAP_ON
            #pragma multi_compile_fog
            #pragma exclude_renderers metal xbox360 xboxone ps3 ps4 psp2 
            #pragma target 3.0
            uniform sampler2D _BaseTexture; uniform float4 _BaseTexture_ST;
            uniform float4 _SpecColGlossA;
            uniform sampler2D _NormalMap; uniform float4 _NormalMap_ST;
            uniform float4 _BaseColourOpacA;
            uniform sampler2D _Texture_SpecColorGlossA; uniform float4 _Texture_SpecColorGlossA_ST;
            uniform float _AdditonalGradient;
            uniform sampler2D _OnTopTexturemaskA;
            uniform float _MaskTiling;
            uniform float _TopTextureTiling;
            uniform float _TopTextureAmount;
            uniform float4 _TopTextureColour;
            struct VertexInput {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float4 tangent : TANGENT;
                float2 texcoord0 : TEXCOORD0;
                float2 texcoord1 : TEXCOORD1;
                float2 texcoord2 : TEXCOORD2;
            };
            struct VertexOutput {
                float4 pos : SV_POSITION;
                float2 uv0 : TEXCOORD0;
                float2 uv1 : TEXCOORD1;
                float2 uv2 : TEXCOORD2;
                float4 posWorld : TEXCOORD3;
                float3 normalDir : TEXCOORD4;
                float3 tangentDir : TEXCOORD5;
                float3 bitangentDir : TEXCOORD6;
                LIGHTING_COORDS(7,8)
                UNITY_FOG_COORDS(9)
            };
            VertexOutput vert (VertexInput v) {
                VertexOutput o = (VertexOutput)0;
                o.uv0 = v.texcoord0;
                o.uv1 = v.texcoord1;
                o.uv2 = v.texcoord2;
                o.normalDir = UnityObjectToWorldNormal(v.normal);
                o.tangentDir = normalize( mul( unity_ObjectToWorld, float4( v.tangent.xyz, 0.0 ) ).xyz );
                o.bitangentDir = normalize(cross(o.normalDir, o.tangentDir) * v.tangent.w);
                float3 recipObjScale = float3( length(unity_WorldToObject[0].xyz), length(unity_WorldToObject[1].xyz), length(unity_WorldToObject[2].xyz) );
                float3 objScale = 1.0/recipObjScale;
                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
                float3 lightColor = _LightColor0.rgb;
                o.pos = mul(UNITY_MATRIX_MVP, v.vertex );
                UNITY_TRANSFER_FOG(o,o.pos);
                TRANSFER_VERTEX_TO_FRAGMENT(o)
                return o;
            }
            float4 frag(VertexOutput i) : COLOR {
                float3 recipObjScale = float3( length(unity_WorldToObject[0].xyz), length(unity_WorldToObject[1].xyz), length(unity_WorldToObject[2].xyz) );
                float3 objScale = 1.0/recipObjScale;
                i.normalDir = normalize(i.normalDir);
                float3x3 tangentTransform = float3x3( i.tangentDir, i.bitangentDir, i.normalDir);
                float3 viewDirection = normalize(_WorldSpaceCameraPos.xyz - i.posWorld.xyz);
                float3 _NormalMap_var = UnpackNormal(tex2D(_NormalMap,TRANSFORM_TEX(i.uv0, _NormalMap)));
                float3 normalLocal = (_NormalMap_var.rgb*float3(3,3,1));
                float3 normalDirection = normalize(mul( normalLocal, tangentTransform )); // Perturbed normals
                clip(_BaseColourOpacA.a - 0.5);
                float3 lightDirection = normalize(lerp(_WorldSpaceLightPos0.xyz, _WorldSpaceLightPos0.xyz - i.posWorld.xyz,_WorldSpaceLightPos0.w));
                float3 lightColor = _LightColor0.rgb;
                float3 halfDirection = normalize(viewDirection+lightDirection);
////// Lighting:
                float attenuation = LIGHT_ATTENUATION(i);
                float3 attenColor = attenuation * _LightColor0.xyz;
                float Pi = 3.141592654;
                float InvPi = 0.31830988618;
///////// Gloss:
                float node_4118 = i.normalDir.g;
                float node_3239 = clamp(((_AdditonalGradient+(_AdditonalGradient*(node_4118/objScale.g)))+node_4118),0.5,3);
                float4 _Texture_SpecColorGlossA_var = tex2D(_Texture_SpecColorGlossA,TRANSFORM_TEX(i.uv0, _Texture_SpecColorGlossA));
                float gloss = (node_3239*(_SpecColGlossA.a*_Texture_SpecColorGlossA_var.a));
                float specPow = exp2( gloss * 10.0+1.0);
////// Specular:
                float NdotL = max(0, dot( normalDirection, lightDirection ));
                float LdotH = max(0.0,dot(lightDirection, halfDirection));
                float3 specularColor = (node_3239*(_SpecColGlossA.rgb*_Texture_SpecColorGlossA_var.rgb));
                float specularMonochrome;
                float4 _BaseTexture_var = tex2D(_BaseTexture,TRANSFORM_TEX(i.uv0, _BaseTexture));
                float2 node_3825 = (float2(_TopTextureTiling,_TopTextureTiling)*i.uv0);
                float4 node_4753 = tex2D(_OnTopTexturemaskA,node_3825);
                float2 node_9514 = (i.uv0*float2(_MaskTiling,_MaskTiling));
                float4 node_2083 = tex2D(_OnTopTexturemaskA,node_9514);
                float3 diffuseColor = lerp(((node_3239*(_BaseColourOpacA.rgb*_BaseTexture_var.rgb))*((_AdditonalGradient*_NormalMap_var.b)*clamp(_NormalMap_var.g,0.5,1))),(_TopTextureColour.rgb*node_4753.rgb),saturate((node_2083.a*(node_4118*_TopTextureAmount)))); // Need this for specular when using metallic
                diffuseColor = EnergyConservationBetweenDiffuseAndSpecular(diffuseColor, specularColor, specularMonochrome);
                specularMonochrome = 1.0-specularMonochrome;
                float NdotV = max(0.0,dot( normalDirection, viewDirection ));
                float NdotH = max(0.0,dot( normalDirection, halfDirection ));
                float VdotH = max(0.0,dot( viewDirection, halfDirection ));
                float visTerm = SmithJointGGXVisibilityTerm( NdotL, NdotV, 1.0-gloss );
                float normTerm = max(0.0, GGXTerm(NdotH, 1.0-gloss));
                float specularPBL = (NdotL*visTerm*normTerm) * (UNITY_PI / 4);
                if (IsGammaSpace())
                    specularPBL = sqrt(max(1e-4h, specularPBL));
                specularPBL = max(0, specularPBL * NdotL);
                float3 directSpecular = attenColor*specularPBL*FresnelTerm(specularColor, LdotH);
                float3 specular = directSpecular;
/////// Diffuse:
                NdotL = max(0.0,dot( normalDirection, lightDirection ));
                half fd90 = 0.5 + 2 * LdotH * LdotH * (1-gloss);
                float nlPow5 = Pow5(1-NdotL);
                float nvPow5 = Pow5(1-NdotV);
                float3 directDiffuse = ((1 +(fd90 - 1)*nlPow5) * (1 + (fd90 - 1)*nvPow5) * NdotL) * attenColor;
                diffuseColor *= 1-specularMonochrome;
                float3 diffuse = directDiffuse * diffuseColor;
/// Final Color:
                float3 finalColor = diffuse + specular;
                fixed4 finalRGBA = fixed4(finalColor * 1,0);
                UNITY_APPLY_FOG(i.fogCoord, finalRGBA);
                return finalRGBA;
            }
            ENDCG
        }
        Pass {
            Name "ShadowCaster"
            Tags {
                "LightMode"="ShadowCaster"
            }
            Offset 1, 1
            
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #define UNITY_PASS_SHADOWCASTER
            #define SHOULD_SAMPLE_SH ( defined (LIGHTMAP_OFF) && defined(DYNAMICLIGHTMAP_OFF) )
            #define _GLOSSYENV 1
            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "UnityPBSLighting.cginc"
            #include "UnityStandardBRDF.cginc"
            #pragma fragmentoption ARB_precision_hint_fastest
            #pragma multi_compile_shadowcaster
            #pragma multi_compile LIGHTMAP_OFF LIGHTMAP_ON
            #pragma multi_compile DIRLIGHTMAP_OFF DIRLIGHTMAP_COMBINED DIRLIGHTMAP_SEPARATE
            #pragma multi_compile DYNAMICLIGHTMAP_OFF DYNAMICLIGHTMAP_ON
            #pragma multi_compile_fog
            #pragma exclude_renderers metal xbox360 xboxone ps3 ps4 psp2 
            #pragma target 3.0
            uniform float4 _BaseColourOpacA;
            struct VertexInput {
                float4 vertex : POSITION;
                float2 texcoord1 : TEXCOORD1;
                float2 texcoord2 : TEXCOORD2;
            };
            struct VertexOutput {
                V2F_SHADOW_CASTER;
                float2 uv1 : TEXCOORD1;
                float2 uv2 : TEXCOORD2;
                float4 posWorld : TEXCOORD3;
            };
            VertexOutput vert (VertexInput v) {
                VertexOutput o = (VertexOutput)0;
                o.uv1 = v.texcoord1;
                o.uv2 = v.texcoord2;
                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
                o.pos = mul(UNITY_MATRIX_MVP, v.vertex );
                TRANSFER_SHADOW_CASTER(o)
                return o;
            }
            float4 frag(VertexOutput i) : COLOR {
                float3 viewDirection = normalize(_WorldSpaceCameraPos.xyz - i.posWorld.xyz);
                clip(_BaseColourOpacA.a - 0.5);
                SHADOW_CASTER_FRAGMENT(i)
            }
            ENDCG
        }
        Pass {
            Name "Meta"
            Tags {
                "LightMode"="Meta"
            }
            Cull Off
            
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #define UNITY_PASS_META 1
            #define SHOULD_SAMPLE_SH ( defined (LIGHTMAP_OFF) && defined(DYNAMICLIGHTMAP_OFF) )
            #define _GLOSSYENV 1
            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "UnityPBSLighting.cginc"
            #include "UnityStandardBRDF.cginc"
            #include "UnityMetaPass.cginc"
            #pragma fragmentoption ARB_precision_hint_fastest
            #pragma multi_compile_shadowcaster
            #pragma multi_compile LIGHTMAP_OFF LIGHTMAP_ON
            #pragma multi_compile DIRLIGHTMAP_OFF DIRLIGHTMAP_COMBINED DIRLIGHTMAP_SEPARATE
            #pragma multi_compile DYNAMICLIGHTMAP_OFF DYNAMICLIGHTMAP_ON
            #pragma multi_compile_fog
            #pragma exclude_renderers metal xbox360 xboxone ps3 ps4 psp2 
            #pragma target 3.0
            uniform sampler2D _BaseTexture; uniform float4 _BaseTexture_ST;
            uniform float4 _SpecColGlossA;
            uniform sampler2D _NormalMap; uniform float4 _NormalMap_ST;
            uniform float4 _BaseColourOpacA;
            uniform sampler2D _Texture_SpecColorGlossA; uniform float4 _Texture_SpecColorGlossA_ST;
            uniform float _AdditonalGradient;
            uniform sampler2D _OnTopTexturemaskA;
            uniform float _MaskTiling;
            uniform float _TopTextureTiling;
            uniform float _TopTextureAmount;
            uniform float4 _TopTextureColour;
            struct VertexInput {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float2 texcoord0 : TEXCOORD0;
                float2 texcoord1 : TEXCOORD1;
                float2 texcoord2 : TEXCOORD2;
            };
            struct VertexOutput {
                float4 pos : SV_POSITION;
                float2 uv0 : TEXCOORD0;
                float2 uv1 : TEXCOORD1;
                float2 uv2 : TEXCOORD2;
                float4 posWorld : TEXCOORD3;
                float3 normalDir : TEXCOORD4;
            };
            VertexOutput vert (VertexInput v) {
                VertexOutput o = (VertexOutput)0;
                o.uv0 = v.texcoord0;
                o.uv1 = v.texcoord1;
                o.uv2 = v.texcoord2;
                o.normalDir = UnityObjectToWorldNormal(v.normal);
                float3 recipObjScale = float3( length(unity_WorldToObject[0].xyz), length(unity_WorldToObject[1].xyz), length(unity_WorldToObject[2].xyz) );
                float3 objScale = 1.0/recipObjScale;
                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
                o.pos = UnityMetaVertexPosition(v.vertex, v.texcoord1.xy, v.texcoord2.xy, unity_LightmapST, unity_DynamicLightmapST );
                return o;
            }
            float4 frag(VertexOutput i) : SV_Target {
                float3 recipObjScale = float3( length(unity_WorldToObject[0].xyz), length(unity_WorldToObject[1].xyz), length(unity_WorldToObject[2].xyz) );
                float3 objScale = 1.0/recipObjScale;
                i.normalDir = normalize(i.normalDir);
                float3 viewDirection = normalize(_WorldSpaceCameraPos.xyz - i.posWorld.xyz);
                float3 normalDirection = i.normalDir;
                UnityMetaInput o;
                UNITY_INITIALIZE_OUTPUT( UnityMetaInput, o );
                
                o.Emission = 0;
                
                float node_4118 = i.normalDir.g;
                float node_3239 = clamp(((_AdditonalGradient+(_AdditonalGradient*(node_4118/objScale.g)))+node_4118),0.5,3);
                float4 _BaseTexture_var = tex2D(_BaseTexture,TRANSFORM_TEX(i.uv0, _BaseTexture));
                float3 _NormalMap_var = UnpackNormal(tex2D(_NormalMap,TRANSFORM_TEX(i.uv0, _NormalMap)));
                float2 node_3825 = (float2(_TopTextureTiling,_TopTextureTiling)*i.uv0);
                float4 node_4753 = tex2D(_OnTopTexturemaskA,node_3825);
                float2 node_9514 = (i.uv0*float2(_MaskTiling,_MaskTiling));
                float4 node_2083 = tex2D(_OnTopTexturemaskA,node_9514);
                float3 diffColor = lerp(((node_3239*(_BaseColourOpacA.rgb*_BaseTexture_var.rgb))*((_AdditonalGradient*_NormalMap_var.b)*clamp(_NormalMap_var.g,0.5,1))),(_TopTextureColour.rgb*node_4753.rgb),saturate((node_2083.a*(node_4118*_TopTextureAmount))));
                float4 _Texture_SpecColorGlossA_var = tex2D(_Texture_SpecColorGlossA,TRANSFORM_TEX(i.uv0, _Texture_SpecColorGlossA));
                float3 specColor = (node_3239*(_SpecColGlossA.rgb*_Texture_SpecColorGlossA_var.rgb));
                float specularMonochrome = max(max(specColor.r, specColor.g),specColor.b);
                diffColor *= (1.0-specularMonochrome);
                float roughness = 1.0 - (node_3239*(_SpecColGlossA.a*_Texture_SpecColorGlossA_var.a));
                o.Albedo = diffColor + specColor * roughness * roughness * 0.5;
                
                return UnityMetaFragment( o );
            }
            ENDCG
        }
    }
    FallBack "Diffuse"
    CustomEditor "ShaderForgeMaterialInspector"
}
